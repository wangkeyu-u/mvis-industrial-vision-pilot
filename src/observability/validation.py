"""Machine-readable stability and non-KPI performance sampling."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from starlette.types import ASGIApp, Message

from src.core.config import ServiceSettings
from src.core.model_registry import ModelRegistry
from src.core.schemas import SCHEMA_VERSION
from src.observability.resources import peak_memory_mb

_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


async def sample_api_stability(
    app: ASGIApp,
    settings: ServiceSettings,
    registry: ModelRegistry,
    *,
    request_count: int = 100,
) -> dict[str, Any]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")

    runtime = registry.runtime_status()
    started = time.perf_counter()
    memory_before = peak_memory_mb()
    wall_samples: list[float] = []
    api_samples: list[int] = []
    statuses: dict[str, int] = {}
    request_ids: set[str] = set()
    errors: dict[str, int] = {}

    image = "data:image/png;base64," + base64.b64encode(_ONE_PIXEL_PNG).decode()
    for index in range(request_count):
        request_id = f"req_stability_{index:04d}"
        body = json.dumps(
            {
                "image": image,
                "query": "确认合规",
                "task": "inspect",
                "model": "active",
                "use_specialist": False,
                "options": {"temperature": 0, "seed": 42},
            },
            separators=(",", ":"),
        ).encode()
        wall_started = time.perf_counter()
        status, response_headers, payload = await _asgi_json_request(
            app, body, request_id
        )
        wall_samples.append((time.perf_counter() - wall_started) * 1000)
        statuses[str(status)] = statuses.get(str(status), 0) + 1
        returned_id = payload.get("request_id") or payload.get("error", {}).get(
            "request_id"
        )
        header_id = response_headers.get("x-request-id")
        if isinstance(returned_id, str) and returned_id == header_id:
            request_ids.add(returned_id)
        if status == 200 and isinstance(payload.get("latency_ms"), int):
            api_samples.append(payload["latency_ms"])
        elif status != 200:
            error_code = str(payload.get("error", {}).get("code", "UNKNOWN"))
            errors[error_code] = errors.get(error_code, 0) + 1

    memory_after = peak_memory_mb()
    success_count = statuses.get("200", 0)
    selected_mode = str(runtime["selected_mode"])
    return {
        "report_type": "mvis_stability_sample",
        "report_version": "1.0.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": settings.service_name,
        "service_version": settings.service_version,
        "code_version": settings.code_version,
        "api_schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "workload": {
            "transport": "in_process_asgi",
            "request_count": request_count,
            "concurrency": 1,
            "image": "embedded_1x1_png",
            "query": "fixed_compliant_probe",
        },
        "stability": {
            "success_count": success_count,
            "success_rate": round(success_count / request_count, 6),
            "unique_request_ids": len(request_ids),
            "status_counts": statuses,
            "error_counts": errors,
        },
        "performance_sample": {
            "wall_latency_ms": _summary(wall_samples),
            "api_latency_ms": _summary([float(value) for value in api_samples]),
            "total_duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "process_peak_memory_before_mb": memory_before,
            "process_peak_memory_after_mb": memory_after,
            "memory_budget_mb": settings.memory_budget_mb,
            "kpi_eligible": False,
            "kpi_exclusion_reason": (
                "mock_latency_is_not_real_model_kpi"
                if selected_mode == "mock"
                else "in_process_smoke_sample_is_not_hardware_benchmark"
            ),
        },
    }


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


async def _asgi_json_request(
    app: ASGIApp, body: bytes, request_id: str
) -> tuple[int, dict[str, str], dict[str, Any]]:
    first_receive = True
    outbound: list[Message] = []

    async def receive() -> Message:
        nonlocal first_receive
        if first_receive:
            first_receive = False
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        outbound.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/analyze",
        "raw_path": b"/v1/analyze",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"preflight"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-request-id", request_id.encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8001),
    }
    await app(scope, receive, send)  # type: ignore[arg-type]
    start = next(item for item in outbound if item["type"] == "http.response.start")
    response_body = b"".join(
        item.get("body", b"")
        for item in outbound
        if item["type"] == "http.response.body"
    )
    headers = {
        name.decode().lower(): value.decode() for name, value in start["headers"]
    }
    return start["status"], headers, json.loads(response_body)


def _summary(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "mean": round(fmean(ordered), 3),
        "p50": round(ordered[math.ceil(len(ordered) * 0.50) - 1], 3),
        "p95": round(ordered[math.ceil(len(ordered) * 0.95) - 1], 3),
        "max": round(ordered[-1], 3),
    }
