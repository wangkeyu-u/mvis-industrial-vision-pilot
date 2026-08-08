"""Export a self-checking, reproducible offline evaluation package."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .acceptance import AcceptanceThresholds, evaluate_kpi_acceptance
from .confidence import bootstrap_confidence_intervals
from .human_report import render_html_report, render_markdown_report
from .reporting import EvaluationReport, SampleEvaluation

FIXTURE_NOTICE = (
    "This package contains contract fixture metrics. It validates the evaluation "
    "pipeline and must not be reported as model performance."
)
MOCK_NOTICE = "This package contains mock outputs and must not be reported as model performance."
PILOT_NOTICE = (
    "This package uses a pilot dataset whose frozen test split is below the formal KPI "
    "sample floor. Metrics are exploratory and must not be reported as formal KPI acceptance."
)


@dataclass(frozen=True)
class EvaluationPackageExport:
    directory: Path
    component_hashes: Mapping[str, str]
    fixture_only: bool
    mock_only: bool
    pilot_only: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _git_value(arguments: list[str], repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_environment_info(repository: str | Path | None = None) -> dict[str, Any]:
    """Collect runtime and code identity without contacting the network."""

    dependency_versions = {}
    for distribution in ("Pillow", "PyYAML"):
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    repository_path = Path(repository).resolve() if repository is not None else Path.cwd().resolve()
    git_commit = _git_value(["rev-parse", "HEAD"], repository_path)
    git_status = _git_value(["status", "--porcelain", "--untracked-files=normal"], repository_path)
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": dependency_versions,
        "code": {
            "git_commit": git_commit,
            "git_dirty": bool(git_status) if git_status is not None else None,
        },
    }


def _sample_slices(sample: SampleEvaluation) -> set[str]:
    return set(sample.truth.slices) | {f"prediction_source:{sample.prediction.source_kind}"}


def build_failure_slices(report: EvaluationReport) -> dict[str, Any]:
    by_code: dict[str, dict[str, Any]] = {}
    for sample in report.failure_cases:
        for code in sample.failure_codes:
            entry = by_code.setdefault(code, {"count": 0, "sample_ids": []})
            entry["count"] += 1
            entry["sample_ids"].append(sample.sample_id)
    by_slice = {}
    slice_names = sorted({name for sample in report.samples for name in _sample_slices(sample)})
    for name in slice_names:
        members = [sample for sample in report.samples if name in _sample_slices(sample)]
        failed = [sample for sample in members if sample.failure_codes]
        codes: dict[str, int] = {}
        for sample in failed:
            for code in sample.failure_codes:
                codes[code] = codes.get(code, 0) + 1
        by_slice[name] = {
            "sample_count": len(members),
            "failure_count": len(failed),
            "failure_rate": len(failed) / len(members) if members else 0.0,
            "failure_codes": dict(sorted(codes.items())),
            "failure_sample_ids": [sample.sample_id for sample in failed],
        }
    return {
        "failure_case_count": len(report.failure_cases),
        "by_code": dict(sorted(by_code.items())),
        "by_slice": by_slice,
    }


def export_evaluation_package(
    report: EvaluationReport,
    output_directory: str | Path,
    *,
    resolved_config: Mapping[str, Any],
    data_manifest_path: str | Path,
    created_at: str,
    baseline_metrics: Mapping[str, int | float] | None = None,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260808,
    fixture_only: bool = False,
    mock_only: bool = False,
    comparison: Mapping[str, Any] | None = None,
    repository: str | Path | None = None,
) -> EvaluationPackageExport:
    """Create a new package directory atomically; existing targets are refused."""

    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if parsed_created_at.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"evaluation package already exists: {destination}")
    manifest_path = Path(data_manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"data manifest does not exist: {manifest_path}")
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("data manifest must contain valid JSON") from exc
    if not isinstance(manifest_value, Mapping):
        raise ValueError("data manifest must be a JSON object")
    manifest_statistics = manifest_value.get("statistics", {})
    if not isinstance(manifest_statistics, Mapping):
        raise ValueError("data manifest statistics must be an object")
    pilot_only = manifest_statistics.get("formal_kpi_eligible") is False

    intervals = bootstrap_confidence_intervals(
        report,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    acceptance = evaluate_kpi_acceptance(
        report.summary,
        baseline_metrics=baseline_metrics,
        thresholds=thresholds,
        fixture_only=fixture_only,
        mock_only=mock_only,
        pilot_only=pilot_only,
    )
    config = dict(resolved_config) | {
        "confidence": {
            "method": "paired_percentile_bootstrap",
            "level": confidence_level,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
        },
        "kpi_thresholds": thresholds.to_dict(),
        "fixture_only": fixture_only,
        "mock_only": mock_only,
        "pilot_only": pilot_only,
    }
    data_provenance = {
        "manifest_filename": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
        "dataset_version": manifest_value.get("dataset_version"),
        "schema_version": manifest_value.get("schema_version"),
        "frozen_test": manifest_value.get("frozen_test"),
        "evaluation_status": manifest_statistics.get("evaluation_status"),
        "formal_kpi_eligible": manifest_statistics.get("formal_kpi_eligible"),
        "formal_kpi_ineligibility_reason": manifest_statistics.get(
            "formal_kpi_ineligibility_reason"
        ),
    }
    score_notice = (
        FIXTURE_NOTICE
        if fixture_only
        else MOCK_NOTICE
        if mock_only
        else PILOT_NOTICE
        if pilot_only
        else None
    )
    metrics = {
        "fixture_only": fixture_only,
        "mock_only": mock_only,
        "pilot_only": pilot_only,
        "fixture_notice": score_notice,
        "summary": dict(report.summary),
        "confidence_intervals": {
            name: interval.to_dict() for name, interval in sorted(intervals.items())
        },
        "kpi_acceptance": acceptance.to_dict(),
    }
    failures = build_failure_slices(report)
    json_components = {
        "config.json": config,
        "data_provenance.json": data_provenance,
        "environment.json": collect_environment_info(repository),
        "failures.json": failures,
        "metrics.json": metrics,
        "report.json": report.to_dict(),
    }
    if comparison is not None:
        json_components["comparison.json"] = dict(comparison)
    text_components = {
        "report.md": render_markdown_report(
            report,
            metrics,
            failures,
            comparison=comparison,
            fixture_only=fixture_only,
            mock_only=mock_only,
            pilot_only=pilot_only,
        ),
        "report.html": render_html_report(
            report,
            metrics,
            failures,
            comparison=comparison,
            fixture_only=fixture_only,
            mock_only=mock_only,
            pilot_only=pilot_only,
        ),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for filename, value in json_components.items():
            _write_json(staging / filename, value)
        for filename, value in text_components.items():
            _write_text(staging / filename, value)
        component_names = sorted([*json_components, *text_components])
        hashes = {filename: _sha256(staging / filename) for filename in component_names}
        _write_json(
            staging / "package_manifest.json",
            {
                "schema_version": "1.0.0",
                "created_at": created_at,
                "fixture_only": fixture_only,
                "mock_only": mock_only,
                "pilot_only": pilot_only,
                "fixture_notice": score_notice,
                "component_sha256": hashes,
            },
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return EvaluationPackageExport(destination, hashes, fixture_only, mock_only, pilot_only)
