"""Strict phase-7 runtime acceptance for VLM, specialist, and fused modes."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from src.core.config import ServiceSettings
from src.core.model_registry import AdapterRequest, ModelAdapter, ModelRegistry
from src.core.preflight import physical_memory_mb, run_preflight
from src.core.registry_factory import build_service_registry
from src.core.schemas import AnalysisMode, ModelIdentity, ModelOutput
from src.core.specialist_runtime import (
    SpecialistAdapter,
    SpecialistRequest,
)
from src.observability.real_validation import _latency_summary, _RunningServer
from src.observability.resources import peak_memory_mb
from src.observability.validation import write_json_report


class _TimeoutOnceModelAdapter:
    def __init__(self, delegate: ModelAdapter) -> None:
        self.delegate = delegate
        self.armed = False

    @property
    def identity(self) -> ModelIdentity:
        return self.delegate.identity

    @property
    def ready(self) -> bool:
        return self.delegate.ready

    async def analyze(self, request: AdapterRequest) -> ModelOutput | Mapping:
        if self.armed:
            self.armed = False
            raise TimeoutError("phase7 controlled model timeout")
        return await self.delegate.analyze(request)


class _TimeoutOnceSpecialistAdapter:
    def __init__(self, delegate: SpecialistAdapter) -> None:
        self.delegate = delegate
        self.armed = False

    @property
    def identity(self) -> ModelIdentity:
        return self.delegate.identity

    @property
    def ready(self) -> bool:
        return self.delegate.ready

    async def detect(self, request: SpecialistRequest):  # type: ignore[no-untyped-def]
        if self.armed:
            self.armed = False
            raise TimeoutError("phase7 controlled specialist timeout")
        return await self.delegate.detect(request)


def run_phase7_release_acceptance(
    settings: ServiceSettings,
    *,
    sample_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 18087,
    warmup_runs: int = 5,
    measured_runs: int = 30,
) -> dict[str, Any]:
    """Measure only strict-real components; mock is never substituted or timed."""

    if settings.model_mode != "real":
        raise ValueError("phase7 acceptance requires MVIS_MODEL_MODE=real")
    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")
    mode = AnalysisMode(settings.default_analysis_mode)
    sample = _sample_metadata(Path(sample_path))
    registry = build_service_registry(settings)
    report = _base_report(settings, mode, sample, warmup_runs, measured_runs)
    report["preflight"] = run_preflight(settings, registry)
    report["registry"] = registry.statuses()
    report["quality_qualification"] = _quality_status(registry, mode)
    report["rollback_qualification"] = _rollback_status(registry)
    blockers = _strict_runtime_blockers(registry, mode)
    if blockers:
        report["blockers"] = blockers
        return report

    timeout_adapter = _install_timeout_adapter(registry, mode)
    from src.api.app import create_app

    application = create_app(settings, registry, emit_config_log=False)
    server = _RunningServer(application, host=host, port=port)
    memory_before = peak_memory_mb()
    started = time.perf_counter()
    try:
        server.start()
        with httpx.Client(
            base_url=f"http://{host}:{port}",
            timeout=max(30.0, settings.inference_timeout_seconds + 10.0),
        ) as client:
            report["probes"]["health"] = _get_json(client, "/health/ready")
            report["probes"]["version"] = _get_json(client, "/version")
            report["probes"]["models"] = _get_json(client, "/v1/models")
            initial = _analyze_once(client, Path(sample_path), mode)
            report["probes"]["initial"] = initial
            if not _is_qualified_response(initial, mode):
                report["blockers"] = ["initial_strict_real_analyze_failed"]
                return _finish(report, started, memory_before)

            warmup = [_analyze_once(client, Path(sample_path), mode) for _ in range(warmup_runs)]
            report["probes"]["warmup"] = warmup
            if not all(_is_qualified_response(item, mode) for item in warmup):
                report["blockers"] = ["strict_real_warmup_failed"]
                return _finish(report, started, memory_before)

            measured = [
                _analyze_once(client, Path(sample_path), mode) for _ in range(measured_runs)
            ]
            report["probes"]["measured"] = measured
            timeout_adapter.armed = True
            report["probes"]["failure_recovery"] = {
                "fault": _analyze_once(client, Path(sample_path), mode),
                "recovery": _analyze_once(client, Path(sample_path), mode),
                "injection": "one-shot service-boundary timeout",
                "included_in_latency": False,
            }
            report["rollback_qualification"] = _exercise_real_rollback(
                registry,
                client,
                Path(sample_path),
                mode,
            )
    except Exception as exc:  # noqa: BLE001 - report safe reproducible class only
        report["blockers"] = [f"release_runner_error:{type(exc).__name__}"]
    finally:
        server.stop()

    measured = report["probes"]["measured"]
    qualified = [item for item in measured if _is_qualified_response(item, mode)]
    recovery = report["probes"]["failure_recovery"] or {}
    failure_ok = (recovery.get("fault") or {}).get("status_code") == 504
    recovery_ok = _is_qualified_response(recovery.get("recovery") or {}, mode)
    memory_after = peak_memory_mb()
    runtime_accepted = bool(
        len(qualified) == measured_runs
        and failure_ok
        and recovery_ok
        and memory_after <= settings.memory_budget_mb
    )
    report["runtime_qualification"] = {
        "accepted": runtime_accepted,
        "strict_real": True,
        "mock_samples_used": False,
        "mode": mode.value,
        "measured_runs": measured_runs,
        "successful_measured_runs": len(qualified),
        "wall_latency_ms": _latency_summary(
            [float(item["wall_latency_ms"]) for item in qualified]
        ),
        "process_peak_memory_before_mb": memory_before,
        "process_peak_memory_after_mb": memory_after,
        "memory_budget_mb": settings.memory_budget_mb,
        "timeout_failure_returned_504": failure_ok,
        "post_failure_recovered": recovery_ok,
    }
    report["quality_qualification"] = _quality_status(registry, mode)
    report["pilot_status"] = "passed" if runtime_accepted else "failed"
    report["production_release_status"] = (
        "passed"
        if runtime_accepted and report["quality_qualification"]["production_ready"]
        else "blocked"
    )
    report["status"] = report["pilot_status"]
    if not runtime_accepted and not report["blockers"]:
        report["blockers"] = ["strict_real_runtime_qualification_failed"]
    return _finish(report, started, memory_before)


def write_phase7_release_reports(
    report: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path
) -> None:
    write_json_report(json_path, report)
    destination = Path(markdown_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(render_phase7_markdown(report), encoding="utf-8")
    temporary.replace(destination)


def render_phase7_markdown(report: dict[str, Any]) -> str:
    runtime = report["runtime_qualification"]
    quality = report["quality_qualification"]
    rollback = report["rollback_qualification"]
    latency = runtime["wall_latency_ms"]
    return "\n".join(
        [
            "# Phase 7 Release Report",
            "",
            f"- Pilot runtime status: `{report['pilot_status']}`",
            f"- Production release status: `{report['production_release_status']}`",
            f"- Analysis mode: `{report['analysis_mode']}`",
            "- Runtime measurements used mock: `false`",
            f"- Generated: `{report['timestamp']}`",
            "",
            "## Strict real runtime",
            "",
            f"- Accepted: `{str(runtime['accepted']).lower()}`",
            f"- Successful requests: `{runtime['successful_measured_runs']}/{runtime['measured_runs']}`",
            f"- P50 / P95: `{latency['p50']} / {latency['p95']} ms`",
            f"- Peak process memory: `{runtime['process_peak_memory_after_mb']} MB`",
            f"- Timeout returned 504: `{str(runtime['timeout_failure_returned_504']).lower()}`",
            f"- Post-timeout recovery: `{str(runtime['post_failure_recovered']).lower()}`",
            "",
            "## Quality and promotion gate",
            "",
            f"- Quality status: `{quality['quality_status']}`",
            f"- Signed evaluation accepted: `{str(quality['quality_accepted']).lower()}`",
            f"- Production ready: `{str(quality['production_ready']).lower()}`",
            "- A pilot may be demonstrated when runtime-ready; production remains blocked without an exact-provenance signed evaluator attestation.",
            "",
            "## Rollback",
            "",
            f"- Real candidate rollback exercised: `{str(rollback['real_candidate_exercised']).lower()}`",
            f"- Status: `{rollback['status']}`",
            f"- Reason: `{rollback['reason']}`",
            "",
            "## Evidence limits",
            "",
            "- Latency and memory qualify only this strict-real local Uvicorn run and sample.",
            "- Controlled timeout recovery is excluded from latency statistics.",
            "- Mock lifecycle evidence, when present, is control-plane evidence only and never a real-model KPI.",
            "- See the algorithm evaluation package for classification/localization quality and dataset limits.",
            "",
            "## Blockers",
            "",
            *(f"- `{item}`" for item in (report.get("blockers") or ["none"])),
            "",
        ]
    )


def _strict_runtime_blockers(registry: ModelRegistry, mode: AnalysisMode) -> list[str]:
    runtime = registry.runtime_status()
    blockers: list[str] = []
    if runtime.get("requested_mode") != "real" or runtime.get("selected_mode") != "real":
        blockers.append("strict_real_registry_not_selected")
    if runtime.get("degraded") is not False:
        blockers.append(str(runtime.get("fallback_reason") or "real_registry_degraded"))
    if not registry.mode_runtime_ready(mode):
        blockers.append(f"analysis_mode_not_ready:{mode.value}")
    active = registry.active()
    if mode in {AnalysisMode.VLM_ONLY, AnalysisMode.FUSED} and (
        active is None or active.adapter.identity.base.startswith("mock-")
    ):
        blockers.append("real_vlm_unavailable")
    specialist = registry.specialist_active()
    if mode in {AnalysisMode.SPECIALIST_ONLY, AnalysisMode.FUSED} and (
        specialist is None
        or specialist.adapter.identity.base.startswith("mock-")
        or "test-double" in specialist.source
    ):
        blockers.append("real_specialist_unavailable")
    return list(dict.fromkeys(blockers))


def _install_timeout_adapter(
    registry: ModelRegistry, mode: AnalysisMode
) -> _TimeoutOnceModelAdapter | _TimeoutOnceSpecialistAdapter:
    if mode in {AnalysisMode.SPECIALIST_ONLY, AnalysisMode.FUSED}:
        specialist = registry.specialist_active()
        assert specialist is not None
        wrapper = _TimeoutOnceSpecialistAdapter(specialist.adapter)
        specialist.adapter = wrapper
        return wrapper
    active = registry.active()
    assert active is not None
    model_wrapper = _TimeoutOnceModelAdapter(active.adapter)
    active.adapter = model_wrapper
    return model_wrapper


def _quality_status(registry: ModelRegistry, mode: AnalysisMode) -> dict[str, Any]:
    return {
        "quality_status": registry.mode_quality_status(mode).value,
        "quality_accepted": registry.mode_quality_accepted(mode),
        "serving_tier": registry.mode_serving_tier(mode).value,
        "production_ready": registry.production_ready(mode),
        "runtime_ready_is_separate": True,
    }


def _rollback_status(registry: ModelRegistry) -> dict[str, Any]:
    statuses = registry.statuses()
    previous = statuses["aliases"].get("previous")
    if previous is None:
        return {
            "status": "blocked",
            "real_candidate_exercised": False,
            "reason": "no_ready_previous_real_candidate",
            "mock_substitution_used": False,
        }
    return {
        "status": "available_not_mutated_by_acceptance",
        "real_candidate_exercised": False,
        "reason": "rollback_requires_explicit_release_authorization",
        "previous": previous,
        "mock_substitution_used": False,
    }


def _exercise_real_rollback(
    registry: ModelRegistry,
    client: httpx.Client,
    sample_path: Path,
    mode: AnalysisMode,
) -> dict[str, Any]:
    statuses = registry.statuses()
    previous_id = statuses["aliases"].get("previous")
    current = registry.active()
    if previous_id is None or current is None:
        return _rollback_status(registry)
    current_id = current.model_id
    current_fingerprint = current.model_fingerprint
    try:
        audit = registry.rollback(
            actor="phase7-release-validation",
            request_id="phase7_real_rollback",
            reason="isolated real candidate rollback probe",
            expected_active_model_id=current_id,
            expected_active_fingerprint=current_fingerprint,
        )
        probe = _analyze_once(client, sample_path, mode)
    except Exception as exc:  # noqa: BLE001 - safe report contains class only
        return {
            "status": "failed",
            "real_candidate_exercised": True,
            "reason": f"rollback_probe_error:{type(exc).__name__}",
            "from": current_id,
            "to": previous_id,
            "mock_substitution_used": False,
        }
    return {
        "status": "passed" if _is_qualified_response(probe, mode) else "failed",
        "real_candidate_exercised": True,
        "reason": "isolated_registry_rollback_and_http_recovery",
        "from": current_id,
        "to": previous_id,
        "audit": audit.public_status(),
        "probe": probe,
        "mock_substitution_used": False,
    }


def _sample_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("release sample must be a regular file")
    raw = path.read_bytes()
    with Image.open(path) as image:
        width, height = image.size
        image_format = (image.format or "").lower()
    if image_format not in {"jpeg", "png", "webp"}:
        raise ValueError("release sample format is unsupported")
    return {
        "name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
        "width": width,
        "height": height,
        "format": image_format,
    }


def _get_json(client: httpx.Client, path: str) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.get(path)
    return {
        "status_code": response.status_code,
        "request_id": response.headers.get("x-request-id"),
        "wall_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "body": response.json(),
    }


def _analyze_once(client: httpx.Client, path: Path, mode: AnalysisMode) -> dict[str, Any]:
    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[path.suffix.lower()]
    started = time.perf_counter()
    with path.open("rb") as handle:
        response = client.post(
            "/v1/analyze",
            data={
                "query": "Inspect for visual compliance anomalies and cite grounded evidence.",
                "task": "inspect",
                "model": "active",
                "analysis_mode": mode.value,
                "options": json.dumps(
                    {"temperature": 0, "seed": 20260808, "max_tokens": 128}
                ),
            },
            files={"image": (path.name, handle, content_type)},
        )
    body = response.json()
    return {
        "status_code": response.status_code,
        "request_id": response.headers.get("x-request-id"),
        "wall_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "analysis_mode": body.get("analysis_mode"),
        "model": body.get("model"),
        "specialist": body.get("specialist"),
        "result": body.get("result"),
        "objects": body.get("objects", []),
        "uncertain": body.get("uncertain"),
        "human_review_required": body.get("human_review_required"),
        "api_latency_ms": body.get("latency_ms"),
        "quality_status": body.get("quality_status"),
        "quality_accepted": body.get("quality_accepted"),
        "warnings": body.get("warnings", []),
        "error": body.get("error"),
    }


def _is_qualified_response(item: Mapping[str, Any], mode: AnalysisMode) -> bool:
    model = item.get("model") or {}
    specialist = item.get("specialist") or {}
    warnings = item.get("warnings") or []
    if item.get("status_code") != 200 or item.get("analysis_mode") != mode.value:
        return False
    if mode in {AnalysisMode.VLM_ONLY, AnalysisMode.FUSED} and str(
        model.get("base", "")
    ).startswith("mock-"):
        return False
    if mode in {AnalysisMode.SPECIALIST_ONLY, AnalysisMode.FUSED} and str(
        specialist.get("specialist_id", "")
    ).startswith("mock-"):
        return False
    return "mock_adapter" not in warnings and "mock_specialist" not in warnings


def _base_report(
    settings: ServiceSettings,
    mode: AnalysisMode,
    sample: dict[str, Any],
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, Any]:
    return {
        "report_type": "mvis_phase7_release_acceptance",
        "report_version": "1.0.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "blocked",
        "pilot_status": "blocked",
        "production_release_status": "blocked",
        "analysis_mode": mode.value,
        "hardware": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "physical_memory_mb": physical_memory_mb(),
        },
        "sample": sample,
        "protocol": {
            "transport": "real_uvicorn_http",
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "concurrency": 1,
            "mock_substitution_allowed": False,
        },
        "preflight": None,
        "registry": None,
        "probes": {
            "health": None,
            "version": None,
            "models": None,
            "initial": None,
            "warmup": [],
            "measured": [],
            "failure_recovery": None,
        },
        "runtime_qualification": {
            "accepted": False,
            "strict_real": True,
            "mock_samples_used": False,
            "mode": mode.value,
            "measured_runs": measured_runs,
            "successful_measured_runs": 0,
            "wall_latency_ms": _latency_summary([]),
            "process_peak_memory_before_mb": None,
            "process_peak_memory_after_mb": None,
            "memory_budget_mb": settings.memory_budget_mb,
            "timeout_failure_returned_504": False,
            "post_failure_recovered": False,
        },
        "quality_qualification": _quality_status_placeholder(),
        "rollback_qualification": {
            "status": "not_run",
            "real_candidate_exercised": False,
            "reason": "runtime_not_ready",
            "mock_substitution_used": False,
        },
        "blockers": [],
    }


def _quality_status_placeholder() -> dict[str, Any]:
    return {
        "quality_status": "unvalidated",
        "quality_accepted": False,
        "serving_tier": "pilot",
        "production_ready": False,
        "runtime_ready_is_separate": True,
    }


def _finish(report: dict[str, Any], started: float, memory_before: float) -> dict[str, Any]:
    report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    runtime = report["runtime_qualification"]
    if runtime["process_peak_memory_before_mb"] is None:
        runtime["process_peak_memory_before_mb"] = memory_before
        runtime["process_peak_memory_after_mb"] = peak_memory_mb()
    return report
