from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from src.api.__main__ import main
from src.api.app import create_app
from src.core.config import ServiceSettings
from src.core.preflight import run_preflight
from src.core.registry_factory import RealAdapterUnavailable, build_service_registry
from src.observability.validation import sample_api_stability, write_json_report


def unavailable_loader(_settings):
    raise RealAdapterUnavailable("mlx_runtime_unavailable")


def configured_settings(**overrides) -> ServiceSettings:
    settings = ServiceSettings.load("configs/service/default.yaml")
    return replace(settings, **overrides)


def test_preflight_auto_degraded_is_servable_but_not_production_ready() -> None:
    settings = configured_settings(model_mode="auto")
    registry = build_service_registry(settings, real_loader=unavailable_loader)
    report = run_preflight(settings, registry)

    assert report["status"] == "degraded"
    assert report["can_serve"] is True
    assert report["production_ready"] is False
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["api_schema"]["status"] == "pass"
    assert checks["disk_budget"]["status"] == "pass"
    assert checks["memory_budget"]["status"] == "pass"
    assert checks["model_readiness"]["status"] == "degraded"


def test_preflight_fails_schema_and_resource_budget_mismatch() -> None:
    settings = configured_settings(
        model_mode="mock",
        api_schema_version="9.9.9",
        min_disk_free_mb=10**12,
        memory_budget_mb=10**9,
    )
    registry = build_service_registry(settings)
    report = run_preflight(settings, registry)
    checks = {item["name"]: item for item in report["checks"]}

    assert report["status"] == "fail"
    assert report["can_serve"] is False
    assert checks["api_schema"]["status"] == "fail"
    assert checks["disk_budget"]["status"] == "fail"
    assert checks["memory_budget"]["status"] == "fail"


def test_stability_report_is_machine_readable_and_excludes_mock_kpi(tmp_path) -> None:
    async def scenario():
        settings = configured_settings(model_mode="mock", log_level="WARNING")
        registry = build_service_registry(settings)
        app = create_app(settings, registry)
        return await sample_api_stability(app, settings, registry, request_count=10)

    report = asyncio.run(scenario())
    output = tmp_path / "report.json"
    write_json_report(output, report)
    reloaded = json.loads(output.read_text(encoding="utf-8"))

    assert reloaded["stability"]["success_count"] == 10
    assert reloaded["stability"]["unique_request_ids"] == 10
    assert reloaded["performance_sample"]["kpi_eligible"] is False
    assert (
        reloaded["performance_sample"]["kpi_exclusion_reason"]
        == "mock_latency_is_not_real_model_kpi"
    )


def test_unified_preflight_entry_prints_clean_json(monkeypatch, capsys) -> None:
    monkeypatch.setenv("MVIS_MODEL_MODE", "mock")
    exit_code = main(["--config", "configs/service/default.yaml", "preflight"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["report_type"] == "mvis_preflight"
    assert payload["can_serve"] is True
