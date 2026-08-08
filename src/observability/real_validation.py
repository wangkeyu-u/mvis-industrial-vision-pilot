"""Real-only runtime acceptance through an actual local Uvicorn server."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import socket
import subprocess
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from PIL import Image

from src.core.config import ServiceSettings
from src.core.model_registry import (
    AdapterRequest,
    ModelAdapter,
)
from src.core.preflight import run_preflight
from src.core.registry_factory import build_service_registry
from src.core.schemas import ModelIdentity, ModelOutput
from src.observability.resources import peak_memory_mb
from src.observability.validation import write_json_report


class TimeoutOnceAdapter:
    """Inject one service-boundary timeout, then delegate to the real adapter."""

    def __init__(self, delegate: ModelAdapter) -> None:
        self.delegate = delegate
        self._armed = False

    @property
    def identity(self) -> ModelIdentity:
        return self.delegate.identity

    @property
    def ready(self) -> bool:
        return self.delegate.ready

    def arm(self) -> None:
        self._armed = True

    async def analyze(self, request: AdapterRequest) -> ModelOutput | Mapping:
        if self._armed:
            self._armed = False
            raise TimeoutError("phase5 controlled timeout injection")
        return await self.delegate.analyze(request)


def run_real_runtime_acceptance(
    settings: ServiceSettings,
    *,
    sample_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 18086,
    warmup_runs: int = 5,
    measured_runs: int = 30,
) -> dict[str, Any]:
    """Run acceptance only when diagnostics and a strict real registry are ready."""

    if settings.model_mode != "real":
        raise ValueError("phase5 acceptance requires model_mode=real")
    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")

    from src.inference.diagnostics import diagnose_model

    diagnostics = diagnose_model(settings.model_config_path)
    report = _base_report(settings, diagnostics, sample_path, warmup_runs, measured_runs)
    if not diagnostics["ready"]:
        report["status"] = "blocked"
        report["blockers"] = list(diagnostics["reasons"])
        return report

    registry = build_service_registry(settings)
    preflight = run_preflight(settings, registry)
    report["preflight"] = preflight
    runtime = registry.runtime_status()
    active = registry.active()
    if (
        not registry.ready()
        or active is None
        or runtime.get("requested_mode") != "real"
        or runtime.get("selected_mode") != "real"
        or runtime.get("degraded") is not False
        or active.adapter.identity.base.startswith("mock-")
    ):
        report["status"] = "blocked"
        report["blockers"] = [
            str(runtime.get("fallback_reason") or "strict_real_registry_not_ready")
        ]
        report["registry"] = registry.statuses()
        return report

    timeout_adapter = TimeoutOnceAdapter(active.adapter)
    active.adapter = timeout_adapter
    report["registry"] = registry.statuses()

    from src.api.app import create_app

    application = create_app(settings, registry, emit_config_log=False)
    server = _RunningServer(application, host=host, port=port)
    memory_before = peak_memory_mb()
    started = time.perf_counter()
    try:
        server.start()
        base_url = f"http://{host}:{port}"
        client_timeout = max(30.0, settings.inference_timeout_seconds + 10.0)
        with httpx.Client(base_url=base_url, timeout=client_timeout) as client:
            health = _request_json(client, "GET", "/health/ready")
            version = _request_json(client, "GET", "/version")
            report["probes"]["health"] = health
            report["probes"]["version"] = version

            initial = _analyze_once(client, Path(sample_path))
            report["probes"]["initial_analyze"] = initial
            if initial["status_code"] != 200 or not _is_real_response(initial):
                report["status"] = "failed"
                report["blockers"] = ["initial_real_analyze_failed"]
                return _finish_report(report, started, memory_before)

            warmup = [_analyze_once(client, Path(sample_path)) for _ in range(warmup_runs)]
            report["probes"]["warmup"] = warmup
            if not all(item["status_code"] == 200 and _is_real_response(item) for item in warmup):
                report["status"] = "failed"
                report["blockers"] = ["real_warmup_probe_failed"]
                return _finish_report(report, started, memory_before)

            measured = [_analyze_once(client, Path(sample_path)) for _ in range(measured_runs)]
            report["probes"]["measured"] = measured

            timeout_adapter.arm()
            timeout_probe = _analyze_once(client, Path(sample_path))
            recovery_probe = _analyze_once(client, Path(sample_path))
            report["probes"]["timeout_recovery"] = {
                "injection": "adapter_timeout_once_control",
                "counts_as_model_performance": False,
                "timeout": timeout_probe,
                "recovery": recovery_probe,
            }
    except Exception as exc:  # noqa: BLE001 - preserve safe, reproducible evidence
        report["status"] = "failed"
        report["blockers"] = [f"acceptance_runner_error:{type(exc).__name__}"]
    finally:
        server.stop()

    successful = [
        item
        for item in report["probes"].get("measured", [])
        if item["status_code"] == 200 and _is_real_response(item)
    ]
    latencies = [float(item["wall_latency_ms"]) for item in successful]
    status_counts: dict[str, int] = {}
    for item in report["probes"].get("measured", []):
        key = str(item["status_code"])
        status_counts[key] = status_counts.get(key, 0) + 1
    timeout_recovery = report["probes"].get("timeout_recovery", {})
    timeout_ok = timeout_recovery.get("timeout", {}).get("status_code") == 504
    recovery_ok = timeout_recovery.get("recovery", {}).get(
        "status_code"
    ) == 200 and _is_real_response(timeout_recovery.get("recovery", {}))
    all_measured_ok = len(successful) == measured_runs
    memory_after = peak_memory_mb()
    memory_ok = memory_after <= settings.memory_budget_mb
    eligible = bool(all_measured_ok and timeout_ok and recovery_ok and memory_ok)
    report["performance"] = {
        "eligible": eligible,
        "qualification": "real_model_runtime" if eligible else "failed_real_runtime",
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "successful_measured_runs": len(successful),
        "status_counts": status_counts,
        "wall_latency_ms": _latency_summary(latencies),
        "process_peak_memory_before_mb": memory_before,
        "process_peak_memory_after_mb": memory_after,
        "memory_budget_mb": settings.memory_budget_mb,
        "memory_within_budget": memory_ok,
        "mock_samples_used": False,
    }
    report["status"] = "passed" if eligible else "failed"
    if not eligible and not report["blockers"]:
        report["blockers"] = [
            reason
            for condition, reason in (
                (all_measured_ok, "measured_requests_failed"),
                (timeout_ok, "controlled_timeout_did_not_return_504"),
                (recovery_ok, "post_timeout_real_request_did_not_recover"),
                (memory_ok, "process_peak_memory_exceeded_budget"),
            )
            if not condition
        ]
    return _finish_report(report, started, memory_before)


def write_real_runtime_reports(
    report: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path
) -> None:
    write_json_report(json_path, report)
    destination = Path(markdown_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(render_real_runtime_markdown(report), encoding="utf-8")
    temporary.replace(destination)


def render_real_runtime_markdown(report: dict[str, Any]) -> str:
    model = report["model"]
    hardware = report["hardware"]
    performance = report["performance"]
    sample = report["sample"]
    initial = report["probes"].get("initial_analyze") or {}
    quality = report["quality_qualification"]
    blockers = report.get("blockers") or []
    lines = [
        "# Phase 5 Real Runtime Report",
        "",
        f"- Status: `{report['status']}`",
        "- Requested runtime: `real`",
        f"- Real ready: `{str(report['runtime_qualification']['real_ready']).lower()}`",
        "- Mock used as real evidence: `false`",
        f"- Generated: `{report['timestamp']}`",
        "",
        "## Hardware and model",
        "",
        f"- Hardware: `{hardware['cpu']}` / `{hardware['memory_mb']:.0f} MB`",
        f"- OS: `{hardware['system']} {hardware['release']}`",
        f"- Python: `{hardware['python']}`",
        f"- Model: `{model['model_id']}`",
        f"- Base model: `{model['base_model_id']}`",
        f"- Revision: `{model['revision']}`",
        f"- Config fingerprint: `{model['config_fingerprint']}`",
        f"- Weight SHA-256: `{model['weight_sha256'] or 'unavailable'}`",
        f"- Quantization: `{model['quantization']}`",
        "",
        "## Sample and protocol",
        "",
        f"- Sample: `{sample['name']}`",
        f"- Sample SHA-256: `{sample['sha256']}`",
        f"- Dimensions: `{sample['width']}x{sample['height']}`",
        f"- Query: `{sample['query']}`",
        f"- Warm-up probes: `{report['protocol']['warmup_runs']}`",
        f"- Measured continuous probes: `{report['protocol']['measured_runs']}`",
        "- Timeout recovery: controlled one-shot adapter timeout followed by a real adapter request; excluded from latency KPI.",
        "",
        "## Runtime qualification",
        "",
        f"- Performance eligible: `{str(performance['eligible']).lower()}`",
        f"- Qualification: `{performance['qualification']}`",
        f"- Successful measured runs: `{performance['successful_measured_runs']}`",
        f"- P50 latency: `{performance['wall_latency_ms']['p50']} ms`",
        f"- P95 latency: `{performance['wall_latency_ms']['p95']} ms`",
        f"- Peak process memory: `{performance['process_peak_memory_after_mb']} MB`",
        f"- Memory budget: `{performance['memory_budget_mb']} MB`",
        "",
        "## Observed service output",
        "",
        f"- HTTP status: `{initial.get('status_code')}`",
        f"- Model identity: `{(initial.get('model') or {}).get('base')}`",
        f"- Result: `{initial.get('result')}`",
        f"- Uncertain: `{initial.get('uncertain')}`",
        f"- Evidence objects: `{len(initial.get('objects') or [])}`",
        f"- Reason: {initial.get('reason') or 'unavailable'}",
        "",
        "## Quality scope",
        "",
        f"- Quality eligible from this runtime protocol: `{str(quality['eligible']).lower()}`",
        f"- Status: `{quality['status']}`",
        f"- External evidence: `{quality['external_evidence']}`",
        "- Runtime latency and memory qualification does not imply visual-compliance quality acceptance.",
        "",
        "## Runtime blockers",
        "",
    ]
    lines.extend([f"- `{reason}`" for reason in blockers] or ["- None"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def _base_report(
    settings: ServiceSettings,
    diagnostics: dict[str, Any],
    sample_path: str | Path,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, Any]:
    sample = _sample_metadata(Path(sample_path))
    model = diagnostics["model"]
    return {
        "report_type": "mvis_phase5_real_runtime",
        "report_version": "1.0.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "blocked",
        "runtime_qualification": {
            "requested_mode": "real",
            "real_ready": bool(diagnostics["ready"]),
            "mock_used": False,
            "auto_fallback_allowed": False,
        },
        "hardware": _hardware_status(diagnostics),
        "model": {
            "alias": model["alias"],
            "model_id": model["model_id"],
            "base_model_id": model["base_model_id"],
            "revision": model["revision"],
            "backend": model["backend"],
            "config_fingerprint": model["config_fingerprint"],
            "quantization": f"{diagnostics['quantization']['bits']}-bit/{diagnostics['quantization']['mode']}",
            "allow_download": model["allow_download"],
            "trust_remote_code": model["trust_remote_code"],
            "weight_sha256": _external_weight_sha256(),
        },
        "diagnostics": {
            "ready": diagnostics["ready"],
            "status": diagnostics["status"],
            "reasons": list(diagnostics["reasons"]),
            "dependencies": diagnostics["dependencies"],
            "cache": {
                key: diagnostics["cache"][key]
                for key in (
                    "source",
                    "exists",
                    "complete",
                    "config_present",
                    "weight_files",
                    "reason",
                )
            },
        },
        "sample": sample,
        "protocol": {
            "transport": "real_uvicorn_http",
            "initial_analyze_runs": 1,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "inference_timeout_seconds": settings.inference_timeout_seconds,
        },
        "preflight": None,
        "registry": None,
        "probes": {
            "health": None,
            "version": None,
            "initial_analyze": None,
            "warmup": [],
            "measured": [],
            "timeout_recovery": None,
        },
        "performance": {
            "eligible": False,
            "qualification": "unavailable",
            "warmup_runs": 0,
            "measured_runs": 0,
            "successful_measured_runs": 0,
            "status_counts": {},
            "wall_latency_ms": _latency_summary([]),
            "process_peak_memory_before_mb": None,
            "process_peak_memory_after_mb": None,
            "memory_budget_mb": settings.memory_budget_mb,
            "memory_within_budget": None,
            "mock_samples_used": False,
        },
        "quality_qualification": _quality_qualification(),
        "blockers": [],
        "interpretation": (
            "Real-model metrics are qualified only after strict real readiness, "
            "Uvicorn HTTP probes, 30 successful measured requests, timeout recovery, "
            "and the configured memory budget all pass. Mock measurements are never substituted."
        ),
    }


def _quality_qualification() -> dict[str, Any]:
    evidence = Path("docs/model/phase5_real_model_report.md")
    status = "not_evaluated_by_runtime_protocol"
    if evidence.is_file():
        try:
            prefix = evidence.read_text(encoding="utf-8")[:4096]
        except OSError:
            prefix = ""
        if "quality_not_accepted" in prefix:
            status = "failed_external_algorithm_probe"
    return {
        "eligible": False,
        "status": status,
        "external_evidence": str(evidence),
    }


def _external_weight_sha256() -> str | None:
    evidence = Path("docs/model/phase5_real_model_report.md")
    try:
        content = evidence.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"权重 SHA-256 \| `([0-9a-f]{64})`", content)
    return match.group(1) if match else None


def _sample_metadata(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    with Image.open(path) as image:
        width, height = image.size
        image_format = (image.format or path.suffix.removeprefix(".")).lower()
    return {
        "name": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "width": width,
        "height": height,
        "format": image_format,
        "query": "找出不符合要求的区域，并给出可核验的证据框。",
        "task": "inspect",
        "options": {"temperature": 0, "seed": 20260808, "max_tokens": 128},
    }


def _hardware_status(diagnostics: dict[str, Any]) -> dict[str, Any]:
    environment = diagnostics["environment"]
    return {
        "cpu": _sysctl_value("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "memory_mb": (environment.get("total_memory_bytes") or 0) / (1024 * 1024),
        "system": environment["system"],
        "release": environment["release"],
        "machine": environment["machine"],
        "python": environment["python"],
    }


def _sysctl_value(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _request_json(client: httpx.Client, method: str, path: str) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.request(method, path)
    return {
        "status_code": response.status_code,
        "request_id": response.headers.get("x-request-id"),
        "wall_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "body": response.json(),
    }


def _analyze_once(client: httpx.Client, sample_path: Path) -> dict[str, Any]:
    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(sample_path.suffix.lower(), "application/octet-stream")
    started = time.perf_counter()
    with sample_path.open("rb") as handle:
        response = client.post(
            "/v1/analyze",
            data={
                "query": "找出不符合要求的区域，并给出可核验的证据框。",
                "task": "inspect",
                "model": "active",
                "use_specialist": "false",
                "options": json.dumps({"temperature": 0, "seed": 20260808, "max_tokens": 128}),
            },
            files={"image": (sample_path.name, handle, content_type)},
        )
    body = response.json()
    return {
        "status_code": response.status_code,
        "request_id": response.headers.get("x-request-id"),
        "wall_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "model": body.get("model"),
        "result": body.get("result"),
        "objects": body.get("objects", []),
        "reason": body.get("reason"),
        "uncertain": body.get("uncertain"),
        "api_latency_ms": body.get("latency_ms"),
        "warnings": body.get("warnings", []),
        "error": body.get("error"),
    }


def _is_real_response(item: Mapping[str, Any]) -> bool:
    model = item.get("model") or {}
    warnings = item.get("warnings") or []
    return bool(
        item.get("status_code") == 200
        and not str(model.get("base", "")).startswith("mock-")
        and "mock_adapter" not in warnings
    )


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(ordered[math.ceil(0.50 * len(ordered)) - 1], 3),
        "p95": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
        "max": round(ordered[-1], 3),
    }


def _finish_report(report: dict[str, Any], started: float, memory_before: float) -> dict[str, Any]:
    report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if report["performance"]["process_peak_memory_before_mb"] is None:
        report["performance"]["process_peak_memory_before_mb"] = memory_before
        report["performance"]["process_peak_memory_after_mb"] = peak_memory_mb()
    return report


class _RunningServer:
    def __init__(self, application: Any, *, host: str, port: int) -> None:
        if port <= 0 or port > 65535:
            raise ValueError("port must be in [1, 65535]")
        self.host = host
        self.port = port
        self.server = uvicorn.Server(
            uvicorn.Config(
                application,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        _ensure_port_available(self.host, self.port)
        self.thread.start()
        deadline = time.monotonic() + 30
        while not self.server.started and self.thread.is_alive():
            if time.monotonic() >= deadline:
                raise TimeoutError("Uvicorn did not start within 30 seconds")
            time.sleep(0.02)
        if not self.server.started:
            raise RuntimeError("Uvicorn exited before startup completed")

    def stop(self) -> None:
        if not self.thread.is_alive():
            return
        self.server.should_exit = True
        self.thread.join(timeout=30)


def _ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
