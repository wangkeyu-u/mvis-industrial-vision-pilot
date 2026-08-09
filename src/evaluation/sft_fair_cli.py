"""Paired zero-shot versus LoRA evaluation on one frozen KSDD SFT test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cli import build_records, load_ground_truth, load_predictions, write_report_atomic
from .comparison import compare_model_reports
from .package import export_evaluation_package
from .reporting import evaluate_offline_records

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: str | Path, name: str) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_run_configs(
    zero: Mapping[str, Any],
    lora: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "run_id",
        "dataset_version",
        "prompt_version",
        "test_jsonl_sha256",
        "base_model_revision",
        "model_config_sha256",
        "generation",
        "adapter",
    }
    for name, value in (("zero-shot", zero), ("LoRA", lora)):
        if set(value) != required:
            raise ValueError(f"{name} run config fields must exactly equal {sorted(required)}")
        if value.get("schema_version") != "1.0.0":
            raise ValueError(f"{name} run config schema_version must be 1.0.0")
        if not isinstance(value.get("run_id"), str) or not str(value["run_id"]).strip():
            raise ValueError(f"{name} run_id must be non-empty")
        for field in ("test_jsonl_sha256", "model_config_sha256"):
            if not isinstance(value.get(field), str) or not SHA256_PATTERN.fullmatch(
                str(value[field])
            ):
                raise ValueError(f"{name} {field} must be lowercase SHA-256")
        if not isinstance(value.get("base_model_revision"), str) or len(
            str(value["base_model_revision"])
        ) < 7:
            raise ValueError(f"{name} base_model_revision must be a pinned revision")
        generation = value.get("generation")
        if not isinstance(generation, Mapping) or set(generation) != {
            "seed",
            "do_sample",
            "temperature",
            "top_p",
            "max_tokens",
        }:
            raise ValueError(f"{name} generation config has unexpected fields")
        if (
            isinstance(generation["seed"], bool)
            or not isinstance(generation["seed"], int)
            or generation["do_sample"] is not False
            or generation["temperature"] != 0.0
            or generation["top_p"] != 1.0
            or isinstance(generation["max_tokens"], bool)
            or not isinstance(generation["max_tokens"], int)
            or generation["max_tokens"] <= 0
        ):
            raise ValueError(f"{name} generation config violates deterministic policy")
    for key in required - {"run_id", "adapter"}:
        if zero[key] != lora[key]:
            raise ValueError(f"fair comparison requires identical {key}")
    prompt_policy = manifest.get("prompt_policy", {})
    file_hashes = manifest.get("file_sha256", {})
    if not isinstance(prompt_policy, Mapping) or not isinstance(file_hashes, Mapping):
        raise ValueError("SFT manifest prompt_policy and file_sha256 are required")
    if zero["dataset_version"] != manifest.get("dataset_version"):
        raise ValueError("run dataset_version differs from SFT manifest")
    if zero["prompt_version"] != manifest.get("prompt_version"):
        raise ValueError("run prompt_version differs from SFT manifest")
    if prompt_policy.get("test_prompt_fixed") is not True:
        raise ValueError("SFT manifest must freeze one test prompt")
    if zero["test_jsonl_sha256"] != file_hashes.get("hf/test.jsonl"):
        raise ValueError("run test_jsonl_sha256 differs from SFT manifest")
    zero_adapter = zero["adapter"]
    lora_adapter = lora["adapter"]
    if not isinstance(zero_adapter, Mapping) or not isinstance(lora_adapter, Mapping):
        raise ValueError("adapter run identity must be an object")
    if zero_adapter != {"enabled": False, "sha256": None}:
        raise ValueError("zero-shot run must disable the adapter and use sha256=null")
    if lora_adapter.get("enabled") is not True or not isinstance(
        lora_adapter.get("sha256"), str
    ):
        raise ValueError("LoRA run must enable an adapter with SHA-256")
    if set(lora_adapter) != {"enabled", "sha256"} or not SHA256_PATTERN.fullmatch(
        str(lora_adapter["sha256"])
    ):
        raise ValueError("LoRA adapter sha256 must be lowercase SHA-256")
    return {
        "same_dataset_version": True,
        "same_prompt_version": True,
        "same_test_jsonl_sha256": True,
        "same_base_model_revision": True,
        "same_model_config_sha256": True,
        "same_generation_config": True,
        "only_difference": "adapter and run_id",
        "zero_shot_adapter_enabled": False,
        "lora_adapter_enabled": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--zero-shot-predictions", required=True)
    parser.add_argument("--lora-predictions", required=True)
    parser.add_argument("--zero-shot-run-config", required=True)
    parser.add_argument("--lora-run-config", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--minimum-reliable-samples", type=int, default=30)
    parser.add_argument("--fixture-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    destination = Path(arguments.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"fair evaluation output already exists: {destination}")
    manifest = _load_object(arguments.data_manifest, "data manifest")
    statistics = manifest.get("statistics", {})
    if not isinstance(statistics, Mapping) or statistics.get("formal_kpi_eligible") is not False:
        raise ValueError("KSDD SFT V1 manifest must be explicitly pilot-only")
    zero_config = _load_object(arguments.zero_shot_run_config, "zero-shot run config")
    lora_config = _load_object(arguments.lora_run_config, "LoRA run config")
    fairness = _validate_run_configs(zero_config, lora_config, manifest)

    truths = load_ground_truth(arguments.ground_truth)
    truth_ids = {truth.sample_id for truth in truths}
    zero_predictions = load_predictions(arguments.zero_shot_predictions)
    lora_predictions = load_predictions(arguments.lora_predictions)
    for name, predictions in (("zero-shot", zero_predictions), ("LoRA", lora_predictions)):
        missing = sorted(truth_ids - set(predictions))
        extra = sorted(set(predictions) - truth_ids)
        if missing or extra:
            raise ValueError(
                f"{name} predictions must exactly cover frozen test IDs; "
                f"missing={missing}, extra={extra}"
            )
    zero_report = evaluate_offline_records(
        build_records(truths, zero_predictions), iou_threshold=arguments.iou_threshold
    )
    lora_report = evaluate_offline_records(
        build_records(truths, lora_predictions), iou_threshold=arguments.iou_threshold
    )
    comparison = compare_model_reports(
        zero_report,
        lora_report,
        resamples=arguments.bootstrap_resamples,
        seed=arguments.bootstrap_seed,
        minimum_reliable_samples=arguments.minimum_reliable_samples,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        write_report_atomic(zero_report, staging / "zero_shot_report.json")
        write_report_atomic(lora_report, staging / "lora_report.json")
        comparison_value = comparison.to_dict()
        (staging / "comparison.json").write_text(
            json.dumps(comparison_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        common = {
            "ground_truth_sha256": _sha256(arguments.ground_truth),
            "data_manifest_sha256": _sha256(arguments.data_manifest),
            "iou_threshold": arguments.iou_threshold,
            "fairness": fairness,
        }
        export_evaluation_package(
            zero_report,
            staging / "zero_shot_package",
            resolved_config=common
            | {
                "run": dict(zero_config),
                "predictions_sha256": _sha256(arguments.zero_shot_predictions),
            },
            data_manifest_path=arguments.data_manifest,
            created_at=arguments.created_at,
            bootstrap_resamples=arguments.bootstrap_resamples,
            bootstrap_seed=arguments.bootstrap_seed,
            fixture_only=arguments.fixture_only,
        )
        export_evaluation_package(
            lora_report,
            staging / "lora_package",
            resolved_config=common
            | {
                "run": dict(lora_config),
                "predictions_sha256": _sha256(arguments.lora_predictions),
            },
            data_manifest_path=arguments.data_manifest,
            created_at=arguments.created_at,
            baseline_metrics=zero_report.summary,
            bootstrap_resamples=arguments.bootstrap_resamples,
            bootstrap_seed=arguments.bootstrap_seed,
            fixture_only=arguments.fixture_only,
            comparison=comparison_value,
        )
        fair_report = {
            "schema_version": "1.0.0",
            "pilot_only": not arguments.fixture_only,
            "fixture_only": arguments.fixture_only,
            "formal_kpi_eligible": False,
            "fair_comparison": True,
            "fairness": fairness,
            "sample_count": len(truths),
            "comparison": comparison_value,
            "zero_shot_metrics": dict(zero_report.summary),
            "lora_metrics": dict(lora_report.summary),
            "notice": (
                "Pilot-only paired comparison; do not report formal KPI acceptance or statistical "
                "significance when any metric is below its reliability floor."
            ),
        }
        (staging / "fair_report.json").write_text(
            json.dumps(fair_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "pilot_only": not arguments.fixture_only,
                "fixture_only": arguments.fixture_only,
                "fair_comparison": True,
                "sample_count": len(truths),
                "conclusion_strength": comparison.conclusion_strength,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
