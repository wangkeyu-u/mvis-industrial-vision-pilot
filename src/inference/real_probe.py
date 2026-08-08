"""Offline real-model probe with explicit outputs and performance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from .base import GenerationBackend, ModelAdapter
from .config import load_model_config
from .contracts import BackendRequest, BackendResponse, GenerationConfig, ModelRequest
from .diagnostics import diagnose_model
from .mlx_backend import MlxVlmBackend
from .performance import process_peak_memory_mb
from .qwen3_vl import Qwen3VLAdapter
from .serialization import model_result_to_dict


@dataclass(slots=True)
class RecordingBackend:
    delegate: GenerationBackend
    last_response: BackendResponse | None = None

    @property
    def name(self) -> str:
        return self.delegate.name

    @property
    def ready(self) -> bool:
        return self.delegate.ready

    def load(self) -> None:
        self.delegate.load()

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.last_response = None
        response = self.delegate.generate(request)
        self.last_response = response
        return response


def probe_adapter(
    adapter: ModelAdapter,
    image_paths: Sequence[str | Path],
    *,
    query: str,
    generation: GenerationConfig,
    raw_response: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("at least one probe image is required")
    started_at = _now()
    load_started = time.perf_counter()
    try:
        adapter.load()
    except Exception as exc:
        return {
            "status": "failed",
            "started_at": started_at,
            "ended_at": _now(),
            "load_latency_ms": round((time.perf_counter() - load_started) * 1000, 3),
            "load_error": _error(exc),
            "probes": [],
            "performance": _unavailable_performance("model_load_failed"),
        }
    load_latency_ms = round((time.perf_counter() - load_started) * 1000, 3)

    probes: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, value in enumerate(image_paths, start=1):
        image_path = Path(value).expanduser().resolve()
        with Image.open(image_path) as image:
            width, height = image.size
        request = ModelRequest(
            image=image_path,
            image_width=width,
            image_height=height,
            query=query,
            request_id=f"real-probe-{index}",
            generation=generation,
        )
        inference_started = time.perf_counter()
        try:
            result = adapter.analyze(request)
            latency_ms = round((time.perf_counter() - inference_started) * 1000, 3)
            entry: dict[str, Any] = {
                "status": "completed",
                "result": model_result_to_dict(result, include_raw_text=True),
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - inference_started) * 1000, 3)
            entry = {
                "status": "failed",
                "error": _error(exc),
                "raw_text": raw_response() if raw_response is not None else None,
            }
        latencies.append(latency_ms)
        probes.append(
            {
                "index": index,
                "image_path": str(image_path),
                "image_sha256": _sha256(image_path),
                "image_width": width,
                "image_height": height,
                "latency_ms": latency_ms,
                "peak_memory_mb": process_peak_memory_mb(),
            }
            | entry
        )

    failures = sum(item["status"] == "failed" for item in probes)
    peak_values = [
        float(item["peak_memory_mb"])
        for item in probes
        if item["peak_memory_mb"] is not None
    ]
    return {
        "status": "completed" if not failures else "completed_with_failures",
        "started_at": started_at,
        "ended_at": _now(),
        "load_latency_ms": load_latency_ms,
        "load_error": None,
        "probes": probes,
        "performance": {
            "status": "recorded",
            "reason": None,
            "attempted_runs": len(probes),
            "successful_runs": len(probes) - failures,
            "failed_runs": failures,
            "p50_latency_ms": _nearest_rank(latencies, 50),
            "p95_latency_ms": _nearest_rank(latencies, 95),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "peak_memory_mb": max(peak_values) if peak_values else None,
        },
    }


def run_real_probe(
    config_path: str | Path,
    image_paths: Sequence[str | Path],
    *,
    query: str,
    max_tokens: int = 128,
) -> dict[str, Any]:
    config = load_model_config(config_path)
    backend = RecordingBackend(MlxVlmBackend(config))
    adapter = Qwen3VLAdapter(config, backend)
    generation = GenerationConfig(
        seed=config.generation.seed,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    result = probe_adapter(
        adapter,
        image_paths,
        query=query,
        generation=generation,
        raw_response=(
            lambda: backend.last_response.text if backend.last_response is not None else None
        ),
    )
    return {
        "schema_version": "1.0",
        "probe_kind": "real_mlx_vlm",
        "config_path": str(Path(config_path).resolve()),
        "diagnostics": diagnose_model(config_path),
        "generation": {
            "seed": generation.seed,
            "do_sample": generation.do_sample,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
        },
        "query": query,
        "run": result,
    }


def write_json_atomic(payload: dict[str, Any], destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _error(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]


def _unavailable_performance(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "attempted_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "peak_memory_mb": None,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = run_real_probe(
        arguments.config,
        arguments.image,
        query=arguments.query,
        max_tokens=arguments.max_tokens,
    )
    write_json_atomic(payload, arguments.output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["run"]["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
