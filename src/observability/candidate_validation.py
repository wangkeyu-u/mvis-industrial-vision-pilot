"""Isolated, no-download LoRA candidate load and single-request probe."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.analyze_service import AnalyzeCommand, AnalyzeService
from src.core.config import ServiceSettings
from src.core.model_registry import ModelRegistry
from src.core.registry_factory import _build_lora_registration, _configured_provenance
from src.core.schemas import AnalyzeOptions, AnalyzeTask, ServingTier
from src.observability.resources import peak_memory_mb


def run_lora_candidate_probe(
    settings: ServiceSettings,
    *,
    sample_path: str | Path | None = None,
    run_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load only the configured LoRA candidate and optionally run one real request."""

    started = time.perf_counter()
    registration = _build_lora_registration(settings, _configured_provenance(settings))
    report: dict[str, Any] = {
        "report_type": "mvis_lora_candidate_probe",
        "report_version": "1.0.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "blocked",
        "mock_used": False,
        "downloads_allowed": False,
        "candidate": registration.public_status(),
        "sample": None,
        "response": None,
        "algorithm_diagnostics": _read_training_diagnostics(run_manifest_path),
        "blockers": [],
    }
    if not registration.adapter.ready:
        reason = getattr(registration.adapter, "reason_code", "candidate_not_ready")
        report["blockers"] = [reason]
        diagnostics = report["algorithm_diagnostics"]
        if isinstance(diagnostics, dict) and diagnostics.get("status") == "resource_limit_exceeded":
            report["blockers"].append("training_resource_limit_exceeded")
        report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        report["process_peak_memory_mb"] = peak_memory_mb()
        return report

    registry = ModelRegistry()
    registry.register(registration)
    registry.validate_candidate(
        registration.model_id,
        actor="candidate-probe",
        request_id="candidate_probe_validate",
        reason="isolated adapter load succeeded",
    )
    registry.activate(
        registration.model_id,
        actor="candidate-probe",
        request_id="candidate_probe_activate",
        reason="isolated pilot probe only",
        serving_tier=ServingTier.PILOT,
    )
    report["candidate"] = registration.public_status()
    report["registry"] = registry.statuses()

    if sample_path is not None:
        sample = Path(sample_path)
        if not sample.is_file():
            report["blockers"] = ["sample_unavailable"]
        else:
            raw = sample.read_bytes()
            content_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(sample.suffix.lower(), "")
            report["sample"] = {
                "name": sample.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "byte_size": len(raw),
            }
            if not content_type:
                report["blockers"] = ["sample_media_type_unsupported"]
            else:
                response, _ = asyncio.run(
                    AnalyzeService(settings, registry).analyze(
                        AnalyzeCommand(
                            image=raw,
                            image_content_type=content_type,
                            query="Inspect for visual compliance defects; return only grounded evidence.",
                            task=AnalyzeTask.INSPECT,
                            model="active",
                            use_specialist=False,
                            options=AnalyzeOptions(
                                temperature=0.0,
                                seed=20260808,
                                max_tokens=512,
                            ),
                        ),
                        "candidate_probe_analyze",
                    )
                )
                report["response"] = response.model_dump(mode="json")

    if not report["blockers"]:
        report["status"] = "passed"
    report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    report["process_peak_memory_mb"] = peak_memory_mb()
    return report


def _read_training_diagnostics(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size > 1024 * 1024
        ):
            return {"status": "invalid", "reason": "run_manifest_invalid"}
        raw = candidate.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid", "reason": "run_manifest_parse_failed"}
    if not isinstance(payload, dict):
        return {"status": "invalid", "reason": "run_manifest_not_object"}
    resources = payload.get("resources") if isinstance(payload.get("resources"), dict) else {}
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    return {
        "manifest_name": candidate.name,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "status": payload.get("status"),
        "run_kind": payload.get("run_kind"),
        "error_type": error.get("type"),
        "error_message": error.get("message"),
        "mlx_peak_allocated_mb": resources.get("mlx_peak_allocated_mb"),
        "process_peak_rss_mb": resources.get("process_peak_rss_mb"),
        "memory_limit_mb": resources.get("memory_limit_mb"),
        "adapter_available": bool(artifacts.get("adapter_path")),
        "adapter_sha256": artifacts.get("adapter_sha256"),
    }
