"""Deterministic startup checks shared by CLI validation and service launch."""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import ServiceSettings
from src.core.model_registry import ModelRegistry
from src.core.schemas import SCHEMA_VERSION
from src.observability.resources import peak_memory_mb


def run_preflight(
    settings: ServiceSettings,
    registry: ModelRegistry,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(workspace or Path.cwd()).resolve()
    checks: list[dict[str, Any]] = []

    service_config = Path(settings.config_path)
    checks.append(
        _check(
            "service_config",
            service_config.is_file(),
            required=True,
            path=_safe_path(service_config, root),
        )
    )

    model_config = Path(settings.model_config_path)
    model_config_required = settings.model_mode in {"auto", "real"}
    checks.append(
        _check(
            "model_config",
            model_config.is_file() or not model_config_required,
            required=model_config_required,
            path=_safe_path(model_config, root),
        )
    )

    schema_matches = settings.api_schema_version == SCHEMA_VERSION
    checks.append(
        _check(
            "api_schema",
            schema_matches,
            required=True,
            configured=settings.api_schema_version,
            implemented=SCHEMA_VERSION,
        )
    )

    disk = shutil.disk_usage(root)
    disk_free_mb = round(disk.free / (1024 * 1024), 2)
    checks.append(
        _check(
            "disk_budget",
            disk_free_mb >= settings.min_disk_free_mb,
            required=True,
            free_mb=disk_free_mb,
            minimum_free_mb=settings.min_disk_free_mb,
            path=".",
        )
    )

    memory_peak = peak_memory_mb()
    physical_memory = physical_memory_mb()
    memory_ok = memory_peak <= settings.memory_budget_mb
    if physical_memory is not None:
        memory_ok = memory_ok and physical_memory >= settings.memory_budget_mb
    checks.append(
        _check(
            "memory_budget",
            memory_ok,
            required=True,
            process_peak_mb=memory_peak,
            budget_mb=settings.memory_budget_mb,
            physical_memory_mb=physical_memory,
            note="process peak is preflight-only, not model inference peak",
        )
    )

    runtime = registry.runtime_status()
    model_ready = registry.ready()
    model_check = _check(
        "model_readiness",
        model_ready,
        required=True,
        runtime=runtime,
        active_model=(
            registry.active().public_status() if registry.active() is not None else None
        ),
    )
    if model_ready and runtime["degraded"]:
        model_check["status"] = "degraded"
    checks.append(model_check)

    required_failures = [
        item for item in checks if item["required"] and item["status"] == "fail"
    ]
    degraded = any(item["status"] == "degraded" for item in checks)
    can_serve = not required_failures
    production_ready = bool(
        can_serve
        and not degraded
        and runtime["selected_mode"] == "real"
        and not runtime["degraded"]
    )
    return {
        "report_type": "mvis_preflight",
        "report_version": "1.0.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": settings.service_name,
        "service_version": settings.service_version,
        "code_version": settings.code_version,
        "api_schema_version": SCHEMA_VERSION,
        "status": ("fail" if not can_serve else "degraded" if degraded else "pass"),
        "can_serve": can_serve,
        "production_ready": production_ready,
        "runtime": runtime,
        "checks": checks,
    }


def physical_memory_mb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return round((pages * page_size) / (1024 * 1024), 2)


def _check(
    name: str, passed: bool, *, required: bool, **details: Any
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "required": required,
        "details": details,
    }


def _safe_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return path.name
