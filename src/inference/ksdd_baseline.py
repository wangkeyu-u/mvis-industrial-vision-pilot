"""Frozen KSDD V0 full-test zero-shot runner for local MLX-VLM models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from .base import ModelAdapter
from .config import load_model_config
from .contracts import GenerationConfig, ModelRequest
from .diagnostics import diagnose_model
from .mlx_backend import MlxVlmBackend
from .performance import process_peak_memory_mb
from .qwen3_vl import Qwen3VLAdapter
from .real_probe import RecordingBackend, write_json_atomic
from .serialization import model_result_to_dict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "configs/models/ksdd_zero_shot_prompt_v1.json"
DEFAULT_MANIFEST = ROOT / "data/processed/ksdd_v0/manifest.json"
DEFAULT_SAMPLES = ROOT / "data/processed/ksdd_v0/samples.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase6/zero_shot"


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    schema_version: str
    experiment_id: str
    prompt_id: str
    dataset_version: str
    dataset_manifest_sha256: str
    expected_test_samples: int
    model_config: str
    model_id: str
    model_revision: str
    query: str
    generation: GenerationConfig

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported baseline schema_version: {self.schema_version}")
        for name in (
            "experiment_id",
            "prompt_id",
            "dataset_version",
            "model_config",
            "model_id",
            "model_revision",
            "query",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if len(self.dataset_manifest_sha256) != 64:
            raise ValueError("dataset_manifest_sha256 must be a SHA-256 hex digest")
        if self.expected_test_samples <= 0:
            raise ValueError("expected_test_samples must be positive")
        if not self.generation.deterministic:
            raise ValueError("frozen zero-shot baseline must be deterministic")

    @property
    def query_sha256(self) -> str:
        return hashlib.sha256(self.query.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BaselineSample:
    sample_id: str
    image_path: Path
    image_sha256: str
    image_width: int
    image_height: int


def load_baseline_spec(path: str | Path) -> BaselineSpec:
    payload = _read_json_object(Path(path))
    required = {
        "schema_version",
        "experiment_id",
        "prompt_id",
        "dataset_version",
        "dataset_manifest_sha256",
        "expected_test_samples",
        "model_config",
        "model_id",
        "model_revision",
        "query",
        "generation",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing:
        raise ValueError(f"missing baseline spec fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown baseline spec fields: {', '.join(unknown)}")
    if not isinstance(payload["generation"], Mapping):
        raise ValueError("generation must be a JSON object")
    expected_generation = {"seed", "do_sample", "temperature", "top_p", "max_tokens"}
    if set(payload["generation"]) != expected_generation:
        raise ValueError("generation must contain exactly the deterministic control fields")
    return BaselineSpec(
        schema_version=_string(payload, "schema_version"),
        experiment_id=_string(payload, "experiment_id"),
        prompt_id=_string(payload, "prompt_id"),
        dataset_version=_string(payload, "dataset_version"),
        dataset_manifest_sha256=_string(payload, "dataset_manifest_sha256"),
        expected_test_samples=_integer(payload, "expected_test_samples"),
        model_config=_string(payload, "model_config"),
        model_id=_string(payload, "model_id"),
        model_revision=_string(payload, "model_revision"),
        query=_string(payload, "query"),
        generation=GenerationConfig(**dict(payload["generation"])),
    )


def load_frozen_test_samples(
    spec: BaselineSpec,
    manifest_path: str | Path,
    samples_path: str | Path,
    *,
    verify_images: bool = True,
) -> tuple[BaselineSample, ...]:
    manifest_file = Path(manifest_path).resolve()
    if _sha256(manifest_file) != spec.dataset_manifest_sha256:
        raise ValueError("frozen dataset manifest SHA-256 does not match baseline spec")
    manifest = _read_json_object(manifest_file)
    if manifest.get("dataset_version") != spec.dataset_version:
        raise ValueError("dataset version does not match baseline spec")
    if manifest.get("frozen_test") is not True:
        raise ValueError("dataset manifest does not mark the test split frozen")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("dataset manifest entries must be an array")
    test_entries = [entry for entry in entries if entry.get("split") == "test"]
    if len(test_entries) != spec.expected_test_samples:
        raise ValueError(
            f"frozen test has {len(test_entries)} samples; expected {spec.expected_test_samples}"
        )

    records = {
        record["sample_id"]: record for record in _read_jsonl(Path(samples_path).resolve())
    }
    dataset_root = manifest_file.parent
    result: list[BaselineSample] = []
    identifiers: set[str] = set()
    for entry in test_entries:
        sample_id = entry.get("sample_id")
        relative_image = entry.get("image")
        expected_hash = entry.get("sha256")
        if not all(isinstance(value, str) and value for value in (sample_id, relative_image, expected_hash)):
            raise ValueError("test manifest entry is missing sample_id, image, or sha256")
        if sample_id in identifiers:
            raise ValueError(f"duplicate frozen test sample_id: {sample_id}")
        identifiers.add(sample_id)
        record = records.get(sample_id)
        if record is None or record.get("metadata", {}).get("split") != "test":
            raise ValueError(f"sample record missing or split mismatch: {sample_id}")
        if record.get("image") != relative_image:
            raise ValueError(f"image path differs between manifest and samples: {sample_id}")
        image_path = (dataset_root / relative_image).resolve()
        if not image_path.is_file():
            raise ValueError(f"frozen test image is missing: {sample_id}")
        if verify_images and _sha256(image_path) != expected_hash:
            raise ValueError(f"frozen test image SHA-256 mismatch: {sample_id}")
        with Image.open(image_path) as image:
            width, height = image.size
        result.append(BaselineSample(sample_id, image_path, expected_hash, width, height))
    return tuple(result)


def run_baseline(
    adapter: ModelAdapter,
    samples: Sequence[BaselineSample],
    spec: BaselineSpec,
    output_dir: str | Path,
    *,
    raw_response: Callable[[], str | None] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    spec_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if len(samples) != spec.expected_test_samples:
        raise ValueError("run population does not match expected frozen test size")
    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("run sample_id values must be unique")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "predictions.jsonl"
    raw_path = destination / "raw_outputs.jsonl"
    prediction_temp = destination / ".predictions.jsonl.partial"
    raw_temp = destination / ".raw_outputs.jsonl.partial"
    started_at = _now()
    _reset_mlx_peak_memory()
    load_started = time.perf_counter()
    adapter.load()
    load_latency_ms = round((time.perf_counter() - load_started) * 1000.0, 3)

    latencies: list[float] = []
    completed = 0
    unavailable = 0
    sample_metrics: list[dict[str, Any]] = []
    try:
        with prediction_temp.open("w", encoding="utf-8") as predictions, raw_temp.open(
            "w", encoding="utf-8"
        ) as raw_outputs:
            for index, sample in enumerate(samples, start=1):
                request = ModelRequest(
                    image=sample.image_path,
                    image_width=sample.image_width,
                    image_height=sample.image_height,
                    query=spec.query,
                    request_id=f"{spec.experiment_id}:{sample.sample_id}",
                    generation=spec.generation,
                )
                inference_started = time.perf_counter()
                raw_text: str | None = None
                error: dict[str, str] | None = None
                try:
                    result = adapter.analyze(request)
                    latency_ms = round((time.perf_counter() - inference_started) * 1000.0, 3)
                    raw_text = result.raw_text
                    prediction = {
                        "sample_id": sample.sample_id,
                        "source": "model_result",
                        "prediction": model_result_to_dict(result),
                        "status": "completed",
                    }
                    completed += 1
                except Exception as exc:
                    latency_ms = round((time.perf_counter() - inference_started) * 1000.0, 3)
                    raw_text = raw_response() if raw_response is not None else None
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    prediction = {
                        "sample_id": sample.sample_id,
                        "source": "model_result",
                        "prediction_raw": "unavailable",
                        "status": "unavailable",
                        "error": {
                            "type": type(exc).__name__,
                            "message": "prediction unavailable; see raw_outputs.jsonl",
                        },
                    }
                    unavailable += 1
                latencies.append(latency_ms)
                process_peak = process_peak_memory_mb()
                mlx_peak = _mlx_peak_memory_mb()
                sample_metrics.append(
                    {
                        "sample_id": sample.sample_id,
                        "latency_ms": latency_ms,
                        "process_peak_rss_mb": process_peak,
                        "mlx_peak_allocated_mb": mlx_peak,
                        "status": prediction["status"],
                    }
                )
                predictions.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
                raw_outputs.write(
                    json.dumps(
                        {
                            "sample_id": sample.sample_id,
                            "image_sha256": sample.image_sha256,
                            "image_width": sample.image_width,
                            "image_height": sample.image_height,
                            "latency_ms": latency_ms,
                            "raw_text": raw_text,
                            "error": error,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                predictions.flush()
                raw_outputs.flush()
                print(
                    f"[{index}/{len(samples)}] {sample.sample_id} {prediction['status']} {latency_ms:.3f} ms",
                    flush=True,
                )
            for stream in (predictions, raw_outputs):
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(prediction_temp, prediction_path)
        os.replace(raw_temp, raw_path)
    except BaseException:
        # Keep partial files as honest recovery evidence; never publish them as final JSONL.
        raise

    memory_process = [
        value["process_peak_rss_mb"]
        for value in sample_metrics
        if value["process_peak_rss_mb"] is not None
    ]
    memory_mlx = [
        value["mlx_peak_allocated_mb"]
        for value in sample_metrics
        if value["mlx_peak_allocated_mb"] is not None
    ]
    report = {
        "schema_version": "1.0",
        "run_kind": "ksdd_frozen_full_test_zero_shot",
        "status": "completed" if unavailable == 0 else "completed_with_failures",
        "started_at": started_at,
        "ended_at": _now(),
        "experiment_id": spec.experiment_id,
        "prompt_id": spec.prompt_id,
        "prompt_query_sha256": spec.query_sha256,
        "spec_path": str(Path(spec_path).resolve()) if spec_path else None,
        "spec_sha256": _sha256(Path(spec_path).resolve()) if spec_path else None,
        "dataset": {
            "version": spec.dataset_version,
            "manifest_path": str(Path(manifest_path).resolve()) if manifest_path else None,
            "manifest_sha256": spec.dataset_manifest_sha256,
            "frozen_test": True,
            "sample_count": len(samples),
            "sample_ids": sample_ids,
        },
        "model": {
            "model_id": spec.model_id,
            "revision": spec.model_revision,
            "diagnostics": dict(diagnostics) if diagnostics is not None else None,
        },
        "generation": {
            "seed": spec.generation.seed,
            "do_sample": spec.generation.do_sample,
            "temperature": spec.generation.temperature,
            "top_p": spec.generation.top_p,
            "max_tokens": spec.generation.max_tokens,
            "deterministic": spec.generation.deterministic,
        },
        "counts": {
            "total": len(samples),
            "completed": completed,
            "unavailable": unavailable,
        },
        "performance": {
            "status": "recorded",
            "load_latency_ms": load_latency_ms,
            "p50_latency_ms": _nearest_rank(latencies, 50),
            "p95_latency_ms": _nearest_rank(latencies, 95),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "process_peak_rss_mb": max(memory_process) if memory_process else None,
            "mlx_peak_allocated_mb": max(memory_mlx) if memory_mlx else None,
            "memory_scope": "process_peak_rss_and_mlx_allocator_peak; neither is system-wide unified-memory pressure",
            "samples": sample_metrics,
        },
        "artifacts": {
            "predictions_jsonl": str(prediction_path),
            "predictions_sha256": _sha256(prediction_path),
            "raw_outputs_jsonl": str(raw_path),
            "raw_outputs_sha256": _sha256(raw_path),
        },
    }
    write_json_atomic(report, destination / "run_manifest.json")
    return report


def run_real_baseline(
    spec_path: str | Path,
    manifest_path: str | Path,
    samples_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    spec = load_baseline_spec(spec_path)
    model_config_path = (ROOT / spec.model_config).resolve()
    config = load_model_config(model_config_path)
    if config.model_id != spec.model_id or config.revision != spec.model_revision:
        raise ValueError("model config identity differs from frozen baseline spec")
    samples = load_frozen_test_samples(spec, manifest_path, samples_path)
    diagnostics = diagnose_model(model_config_path)
    if not diagnostics["ready"]:
        raise RuntimeError("pinned real model is not ready: " + "; ".join(diagnostics["reasons"]))
    recording_backend = RecordingBackend(MlxVlmBackend(config))
    adapter = Qwen3VLAdapter(config, recording_backend)
    return run_baseline(
        adapter,
        samples,
        spec,
        output_dir,
        raw_response=lambda: (
            recording_backend.last_response.text
            if recording_backend.last_response is not None
            else None
        ),
        diagnostics=diagnostics,
        spec_path=spec_path,
        manifest_path=manifest_path,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line must be an object: {path}:{line_number}")
            records.append(value)
    return records


def _string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _reset_mlx_peak_memory() -> None:
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except (ImportError, AttributeError, RuntimeError):
        return


def _mlx_peak_memory_mb() -> float | None:
    try:
        import mlx.core as mx

        return round(float(mx.get_peak_memory()) / (1024.0 * 1024.0), 3)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    report = run_real_baseline(
        arguments.spec,
        arguments.manifest,
        arguments.samples,
        arguments.output_dir,
    )
    print(json.dumps({"status": report["status"], **report["counts"]}, sort_keys=True))
    return 0 if report["counts"]["unavailable"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
