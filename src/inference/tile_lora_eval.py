"""Evaluate a tiled MLX-VLM LoRA candidate on the frozen KSDD test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_model_config
from .contracts import GenerationConfig
from .ksdd_baseline import BaselineSample, BaselineSpec, run_baseline
from .mlx_backend import MlxVlmBackend
from .qwen3_vl import Qwen3VLAdapter
from .real_probe import RecordingBackend, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]


def aggregate_tile_predictions(
    rows: Sequence[Mapping[str, Any]], tile_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Map tile-space evidence back to source images and require all three tiles."""

    record_by_id = {str(record["sample_id"]): record for record in tile_records}
    row_by_id = {str(row["sample_id"]): row for row in rows}
    if set(record_by_id) != set(row_by_id):
        raise ValueError("tile records and predictions do not have identical sample IDs")
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for tile_id, record in record_by_id.items():
        tile = record.get("tile")
        if not isinstance(tile, Mapping):
            raise ValueError(f"tile metadata is missing: {tile_id}")
        grouped[str(tile["source_sample_id"])].append((record, row_by_id[tile_id]))

    aggregated = []
    for source_id in sorted(grouped):
        items = sorted(grouped[source_id], key=lambda item: str(item[0]["sample_id"]))
        if len(items) != 3:
            raise ValueError(f"source sample must have exactly three tiles: {source_id}")
        failed = [row for _, row in items if row.get("status") != "completed"]
        if failed:
            aggregated.append(
                {
                    "sample_id": source_id,
                    "source": "model_result",
                    "prediction_raw": "unavailable",
                    "status": "unavailable",
                    "error": {
                        "type": "TilePredictionUnavailable",
                        "message": f"{len(failed)} of 3 tile predictions unavailable",
                    },
                }
            )
            continue

        predictions = [row["prediction"] for _, row in items]
        violations = [prediction for prediction in predictions if prediction["result"] == "violation"]
        uncertainties = [prediction for prediction in predictions if prediction["result"] == "uncertain"]
        first = predictions[0]
        if violations:
            objects = []
            for record, row in items:
                prediction = row["prediction"]
                if prediction["result"] != "violation":
                    continue
                tile = record["tile"]
                output_size = int(tile["output_size"])
                _, y_offset, source_tile_width, y2 = tile["source_bbox_xyxy"]
                source_tile_size = int(source_tile_width)
                scale = source_tile_size / output_size
                for obj in prediction["objects"]:
                    x1, y1, x2, object_y2 = obj["bbox"]
                    bbox = [
                        max(0, math.floor(x1 * scale)),
                        max(0, math.floor(y_offset + y1 * scale)),
                        min(int(tile["source_width"]), math.ceil(x2 * scale)),
                        min(int(tile["source_height"]), math.ceil(y_offset + object_y2 * scale)),
                    ]
                    if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                        objects.append({**obj, "bbox": bbox, "source": "vlm"})
            result = "violation" if objects else "uncertain"
            refusal = None if objects else {
                "code": "insufficient_evidence",
                "message": "Tile violation evidence could not be mapped to the source image.",
                "review_required": True,
            }
            reason = f"Aggregated {len(objects)} mapped evidence objects from three tiles."
        elif uncertainties:
            result = "uncertain"
            objects = []
            refusal = uncertainties[0]["refusal"]
            reason = "At least one of three tile assessments was uncertain."
        else:
            result = "compliant"
            objects = []
            refusal = None
            reason = "All three tiled assessments were compliant."
        prediction = {
            "result": result,
            "objects": objects,
            "reason": reason,
            "uncertain": result == "uncertain",
            "refusal": refusal,
            "provenance": first["provenance"],
            "warnings": ["phase7_three_tile_aggregation"],
        }
        aggregated.append(
            {
                "sample_id": source_id,
                "source": "model_result",
                "prediction": prediction,
                "status": "completed",
            }
        )
    return aggregated


def run_tile_lora_evaluation(
    model_config_path: str | Path,
    adapter_path: str | Path,
    tile_sft_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    tile_root = Path(tile_sft_root).resolve()
    destination = Path(output_dir).resolve()
    adapter_dir = Path(adapter_path).resolve()
    records = _read_jsonl(tile_root / "hf/test.jsonl")
    if len(records) != 168:
        raise ValueError("phase 7 frozen tiled test must contain exactly 168 tiles")
    query = records[0]["messages"][0]["content"][1]["text"]
    if any(record["messages"][0]["content"][1]["text"] != query for record in records):
        raise ValueError("tiled test queries are not frozen identically")
    generation = GenerationConfig(
        seed=20260808, do_sample=False, temperature=0.0, top_p=1.0, max_tokens=128
    )
    config = load_model_config(model_config_path)
    config = replace(
        config,
        adapter_path=str(adapter_dir),
        generation=generation,
        limits={**config.limits, "max_output_tokens": 128},
    )
    samples = tuple(
        BaselineSample(
            sample_id=str(record["sample_id"]),
            image_path=(ROOT / record["images"][0]).resolve(),
            image_sha256=_sha256((ROOT / record["images"][0]).resolve()),
            image_width=384,
            image_height=384,
        )
        for record in records
    )
    tile_manifest = json.loads((tile_root / "sft_manifest.json").read_text(encoding="utf-8"))
    spec = BaselineSpec(
        schema_version="1.0",
        experiment_id="ksdd_phase7_tile_qlora_384_rank4",
        prompt_id="ksdd_tile_prompt_v1",
        dataset_version=str(tile_manifest["dataset_version"]),
        dataset_manifest_sha256=str(tile_manifest["source_manifest_sha256"]),
        expected_test_samples=168,
        model_config=str(model_config_path),
        model_id=config.model_id,
        model_revision=config.revision,
        query=query,
        generation=generation,
    )
    recording = RecordingBackend(MlxVlmBackend(config))
    tile_report = run_baseline(
        Qwen3VLAdapter(config, recording),
        samples,
        spec,
        destination / "tile_runtime",
        raw_response=lambda: recording.last_response.text if recording.last_response else None,
    )
    tile_rows = _read_jsonl(destination / "tile_runtime/predictions.jsonl")
    aggregated = aggregate_tile_predictions(tile_rows, records)
    prediction_path = destination / "predictions.jsonl"
    _write_jsonl_atomic(prediction_path, aggregated)
    tile_latencies = {
        item["sample_id"]: float(item["latency_ms"])
        for item in tile_report["performance"]["samples"]
    }
    source_latencies = [
        sum(tile_latencies[str(record["sample_id"])] for record in records if record["tile"]["source_sample_id"] == source_id)
        for source_id in sorted({str(record["tile"]["source_sample_id"]) for record in records})
    ]
    completed = sum(row["status"] == "completed" for row in aggregated)
    report = {
        "schema_version": "1.0",
        "status": "completed" if completed == 56 else "completed_with_failures",
        "dataset": {
            "frozen_test": True,
            "source_sample_count": 56,
            "tile_sample_count": 168,
            "tile_manifest_sha256": _sha256(tile_root / "sft_manifest.json"),
        },
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "adapter_path": str(adapter_dir),
            "adapter_sha256": _sha256(adapter_dir / "adapters.safetensors"),
        },
        "generation": {
            "seed": generation.seed,
            "max_tokens": generation.max_tokens,
            "deterministic": generation.deterministic,
        },
        "counts": {"total": 56, "completed": completed, "unavailable": 56 - completed},
        "performance": {
            "p50_latency_ms_per_source_image": _nearest_rank(source_latencies, 50),
            "p95_latency_ms_per_source_image": _nearest_rank(source_latencies, 95),
            "mlx_peak_allocated_mb": tile_report["performance"]["mlx_peak_allocated_mb"],
            "process_peak_rss_mb": tile_report["performance"]["process_peak_rss_mb"],
            "load_latency_ms": tile_report["performance"]["load_latency_ms"],
        },
        "artifacts": {
            "predictions_jsonl": str(prediction_path),
            "predictions_sha256": _sha256(prediction_path),
            "tile_run_manifest": str(destination / "tile_runtime/run_manifest.json"),
        },
    }
    write_json_atomic(report, destination / "run_manifest.json")
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile / 100 * len(ordered))) - 1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--tile-sft-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = run_tile_lora_evaluation(
        arguments.model_config,
        arguments.adapter_path,
        arguments.tile_sft_root,
        arguments.output_dir,
    )
    print(json.dumps({"status": report["status"], **report["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
