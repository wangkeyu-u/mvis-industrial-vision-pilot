from __future__ import annotations

import json
import logging

import pytest

from src.core.config import ServiceSettings
from src.core.model_registry import (
    MockModelAdapter,
    ModelRegistration,
    ModelRegistry,
    ModelState,
)
from src.observability.logging import JsonFormatter


def test_health_version_and_model_registration_status(client) -> None:
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "live"

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["details"]["aliases"]["active"] == "mock-compliance-v0"

    version = client.get("/version")
    assert version.status_code == 200
    assert version.json()["api"] == "v1"
    assert version.json()["schema_version"] == "1.0.0"
    assert version.json()["model"]["base"] == "mock-vlm-0"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["models"][0]["state"] == "active"
    assert models.json()["models"][0]["source"] == "built-in-test-double"
    assert models.json()["models"][0]["quantization"] == "none"


def test_readiness_fails_when_active_adapter_is_not_ready(client_factory) -> None:
    client = client_factory(MockModelAdapter(ready=False))
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_registry_enforces_lifecycle() -> None:
    registry = ModelRegistry()
    registry.register(
        ModelRegistration(
            model_id="candidate-a",
            adapter=MockModelAdapter(),
            state=ModelState.CANDIDATE,
            source="test",
        )
    )
    registry.transition("candidate-a", ModelState.VALIDATED)
    registry.transition("candidate-a", ModelState.ACTIVE)
    assert registry.ready()

    with pytest.raises(ValueError, match="invalid model state transition"):
        registry.transition("candidate-a", ModelState.CANDIDATE)


def test_registry_promotion_preserves_previous_alias() -> None:
    registry = ModelRegistry()
    first = MockModelAdapter()
    second = MockModelAdapter()
    registry.register(ModelRegistration("first", first, ModelState.ACTIVE, source="test"))
    registry.register(ModelRegistration("second", second, ModelState.CANDIDATE, source="test"))
    registry.transition("second", ModelState.VALIDATED)
    registry.transition("second", ModelState.ACTIVE)

    status = registry.statuses()
    assert status["aliases"] == {"active": "second", "previous": "first"}
    states = {item["model_id"]: item["state"] for item in status["models"]}
    assert states == {"first": "validated", "second": "active"}


def test_settings_missing_file_uses_real_defaults(tmp_path) -> None:
    settings = ServiceSettings.load(tmp_path / "does-not-exist.yaml")
    assert settings.service_name == "multimodal-vision-service"
    assert settings.max_image_bytes == 10 * 1024 * 1024
    assert settings.inference_timeout_seconds == 8.0


def test_settings_reject_non_positive_resource_limits() -> None:
    with pytest.raises(ValueError, match="concurrency_limit"):
        ServiceSettings(concurrency_limit=0)


def test_settings_environment_selects_model_and_cors(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MVIS_MODEL_MODE", "real")
    monkeypatch.setenv("MVIS_MODEL_CONFIG", "configs/models/custom.json")
    monkeypatch.setenv("MVIS_CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:5173")

    settings = ServiceSettings.load(tmp_path / "missing.yaml")
    assert settings.model_mode == "real"
    assert settings.model_config_path == "configs/models/custom.json"
    assert settings.cors_allowed_origins == (
        "http://127.0.0.1:8000",
        "http://localhost:5173",
    )


def test_json_log_schema_is_complete_and_does_not_include_sensitive_text() -> None:
    record = logging.LogRecord(
        name="mvis",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="analyze_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "req_test"
    record.model_id = "model-a"
    record.adapter_id = "adapter-a"
    record.quantization = "4-bit"
    record.runtime_mode = "real"
    record.degraded = False
    record.fallback_reason = None
    record.task = "inspect"
    record.image_shape = [24, 32]
    record.latency = {"total_ms": 1}
    record.memory_peak_mb = 10.0
    record.status = 200
    record.error_code = None

    payload = json.loads(JsonFormatter().format(record))
    assert {
        "timestamp",
        "level",
        "request_id",
        "model_id",
        "adapter_id",
        "model_revision",
        "config_fingerprint",
        "config_artifact_sha256",
        "model_fingerprint",
        "weight_hash",
        "adapter_hash",
        "data_version",
        "prompt_version",
        "quality_status",
        "quality_accepted",
        "serving_tier",
        "production_ready",
        "quantization",
        "runtime_mode",
        "degraded",
        "fallback_reason",
        "task",
        "image_shape",
        "latency",
        "memory_peak",
        "memory_peak_mb",
        "status",
        "error_code",
    }.issubset(payload)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "base64" not in serialized
    assert "找出不符合要求的区域" not in serialized
