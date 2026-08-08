"""Batch offline evaluation CLI.

Run with ``python -m src.evaluation.cli --ground-truth ... --predictions ...``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters import (
    NormalizedPrediction,
    normalize_analyze_response,
    normalize_model_result_payload,
)
from .package import export_evaluation_package
from .reporting import (
    EvaluationReport,
    GroundTruthRecord,
    OfflineEvaluationRecord,
    evaluate_offline_records,
)


def _input_identity(path: str | Path) -> dict[str, str]:
    input_path = Path(path)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    return {"filename": input_path.name, "sha256": digest}


def _load_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    values = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL value at {path}:{line_number} must be an object")
            values.append(value)
    return values


def load_ground_truth(path: str | Path) -> tuple[GroundTruthRecord, ...]:
    return tuple(GroundTruthRecord.from_dict(value) for value in _load_jsonl(path))


def _normalize_prediction_entry(value: Mapping[str, Any]) -> NormalizedPrediction:
    source = value.get("source", "api")
    payload = value.get("prediction_raw") if "prediction_raw" in value else value.get("prediction")
    if not isinstance(source, str) or source not in {"api", "model_result"}:
        return NormalizedPrediction.invalid("unknown", f"unsupported prediction source: {source!r}")
    if not isinstance(payload, (str, Mapping)):
        return NormalizedPrediction.invalid(source, "prediction or prediction_raw is required")
    if source == "model_result":
        return normalize_model_result_payload(payload)
    return normalize_analyze_response(payload)


def load_predictions(path: str | Path) -> dict[str, NormalizedPrediction]:
    predictions: dict[str, NormalizedPrediction] = {}
    for value in _load_jsonl(path):
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("prediction entry sample_id must be non-empty")
        if sample_id in predictions:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        predictions[sample_id] = _normalize_prediction_entry(value)
    return predictions


def build_records(
    truths: Sequence[GroundTruthRecord],
    predictions: Mapping[str, NormalizedPrediction],
) -> tuple[OfflineEvaluationRecord, ...]:
    truth_ids = {truth.sample_id for truth in truths}
    unknown = sorted(set(predictions) - truth_ids)
    if unknown:
        raise ValueError(f"predictions contain unknown sample_ids: {unknown}")
    return tuple(
        OfflineEvaluationRecord(
            truth,
            predictions.get(
                truth.sample_id,
                NormalizedPrediction.invalid("missing", "prediction is missing"),
            ),
        )
        for truth in truths
    )


def write_report_atomic(report: EvaluationReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ModelResult/API JSON offline")
    parser.add_argument("--ground-truth", required=True, help="ground-truth JSONL path")
    parser.add_argument("--predictions", required=True, help="prediction JSONL path")
    parser.add_argument("--output", required=True, help="machine-readable report JSON path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--package-dir", help="optional new reproducibility package directory")
    parser.add_argument("--data-manifest", help="required with --package-dir")
    parser.add_argument("--created-at", help="timezone-aware ISO-8601 timestamp for package")
    parser.add_argument("--baseline-metrics", help="optional baseline metric JSON")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="mark contract fixtures as ineligible for model acceptance",
    )
    parser.add_argument("--mock-only", action="store_true", help="watermark mock outputs")
    parser.add_argument("--comparison", help="optional candidate comparison JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    truths = load_ground_truth(arguments.ground_truth)
    predictions = load_predictions(arguments.predictions)
    report = evaluate_offline_records(
        build_records(truths, predictions), iou_threshold=arguments.iou_threshold
    )
    write_report_atomic(report, arguments.output)
    if arguments.package_dir:
        if not arguments.data_manifest or not arguments.created_at:
            raise ValueError("--package-dir requires --data-manifest and --created-at")
        baseline = None
        if arguments.baseline_metrics:
            baseline_value = json.loads(
                Path(arguments.baseline_metrics).read_text(encoding="utf-8")
            )
            if not isinstance(baseline_value, Mapping):
                raise ValueError("baseline metrics must be a JSON object")
            baseline = baseline_value
        comparison = None
        if arguments.comparison:
            comparison_value = json.loads(Path(arguments.comparison).read_text(encoding="utf-8"))
            if not isinstance(comparison_value, Mapping):
                raise ValueError("comparison must be a JSON object")
            comparison = comparison_value
        export_evaluation_package(
            report,
            arguments.package_dir,
            resolved_config={
                "ground_truth": _input_identity(arguments.ground_truth),
                "predictions": _input_identity(arguments.predictions),
                "iou_threshold": arguments.iou_threshold,
                "positive_results": ["violation"],
                "negative_results": ["compliant", "no_violation", "negative"],
                "uncertain_result": "uncertain",
            },
            data_manifest_path=arguments.data_manifest,
            created_at=arguments.created_at,
            baseline_metrics=baseline,
            bootstrap_resamples=arguments.bootstrap_resamples,
            bootstrap_seed=arguments.bootstrap_seed,
            fixture_only=arguments.fixture_only,
            mock_only=arguments.mock_only,
            comparison=comparison,
        )
    print(json.dumps(report.summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
