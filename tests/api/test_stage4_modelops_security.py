from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.__main__ import main
from src.api.app import create_app
from src.core.model_registry import (
    AdapterRequest,
    MockModelAdapter,
    ModelRegistration,
    ModelRegistry,
    ModelState,
    build_mock_registry,
    fingerprint_mapping,
)
from src.core.schemas import ModelOutput
from src.observability.logging import JsonFormatter

from .conftest import image_bytes
from .test_analyze import assert_error, post_image

TOKEN = "stage4-modelops-token-1234"
HEADERS = {
    "X-ModelOps-Token": TOKEN,
    "X-ModelOps-Actor": "test-operator",
}


def registry_with_candidate(*, ready: bool = True) -> ModelRegistry:
    registry = build_mock_registry()
    registry.register(
        ModelRegistration(
            model_id="mock-compliance-v1",
            adapter=MockModelAdapter(
                ready=ready,
                base="mock-vlm-1",
                adapter_id="mock-compliance-v1",
            ),
            state=ModelState.CANDIDATE,
            source="stage4-test-candidate",
            quantization="none",
            config_fingerprint=fingerprint_mapping({"model": "mock-compliance-v1", "version": 1}),
        )
    )
    return registry


def test_registry_enforces_validated_gate_atomic_activation_and_rollback() -> None:
    registry = registry_with_candidate()
    original = registry.active()
    assert original is not None

    registry.validate_candidate(
        "mock-compliance-v1",
        actor="test-operator",
        request_id="req_validate",
        reason="candidate checks passed",
    )
    registry.activate(
        "mock-compliance-v1",
        actor="test-operator",
        request_id="req_activate",
        reason="controlled rollout",
        expected_active_model_id=original.model_id,
        expected_active_fingerprint=original.model_fingerprint,
    )

    activated = registry.statuses()
    assert activated["aliases"] == {
        "active": "mock-compliance-v1",
        "previous": "mock-compliance-v0",
    }
    fingerprints = {item["model_id"]: item["model_fingerprint"] for item in activated["models"]}
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in fingerprints.values())

    registry.rollback(
        actor="test-operator",
        request_id="req_rollback",
        reason="rollback drill",
        expected_active_model_id="mock-compliance-v1",
        expected_active_fingerprint=fingerprints["mock-compliance-v1"],
    )
    rolled_back = registry.statuses()
    assert rolled_back["aliases"] == {
        "active": "mock-compliance-v0",
        "previous": "mock-compliance-v1",
    }
    states = {item["model_id"]: item["state"] for item in rolled_back["models"]}
    assert states == {
        "mock-compliance-v0": "active",
        "mock-compliance-v1": "retired",
    }
    actions = [item["action"] for item in registry.audit_records(limit=20)]
    assert actions[-3:] == ["validate", "activate", "rollback"]


def test_dependency_free_lifecycle_cli_demo(capsys) -> None:
    assert main(["lifecycle-demo"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["uses_real_weights"] is False
    assert report["after_activation"]["aliases"]["active"] == "mock-compliance-v1"
    assert report["after_rollback"]["aliases"]["active"] == "mock-compliance-v0"
    assert [item["action"] for item in report["audit"]][-3:] == [
        "validate",
        "activate",
        "rollback",
    ]


def test_validation_rejects_unready_candidate_and_stale_activation_is_atomic() -> None:
    unready = registry_with_candidate(ready=False)
    with pytest.raises(ValueError, match="must be ready"):
        unready.validate_candidate(
            "mock-compliance-v1",
            actor="operator",
            request_id="req_unready",
            reason="should fail",
        )
    assert unready.statuses()["aliases"]["candidate"] == "mock-compliance-v1"

    registry = registry_with_candidate()
    registry.validate_candidate(
        "mock-compliance-v1",
        actor="operator",
        request_id="req_validate",
        reason="validate",
    )
    before = registry.statuses()
    with pytest.raises(ValueError, match="active model changed"):
        registry.activate(
            "mock-compliance-v1",
            actor="operator",
            request_id="req_stale",
            reason="stale operation",
            expected_active_model_id="not-current",
        )
    assert registry.statuses() == before


def test_modelops_api_is_disabled_by_default_and_requires_constant_time_token(
    settings,
) -> None:
    disabled = TestClient(create_app(settings, registry_with_candidate()))
    response = disabled.post(
        "/v1/models/mock-compliance-v1/validate",
        json={"reason": "validate candidate"},
        headers={"X-ModelOps-Actor": "test-operator"},
    )
    assert_error(response, 503, "MODELOPS_DISABLED")

    enabled_settings = replace(settings, modelops_token=TOKEN)
    enabled = TestClient(create_app(enabled_settings, registry_with_candidate()))
    unauthorized = enabled.post(
        "/v1/models/mock-compliance-v1/validate",
        json={"reason": "validate candidate"},
        headers={
            "X-ModelOps-Token": "incorrect-token-value",
            "X-ModelOps-Actor": "test-operator",
        },
    )
    assert_error(unauthorized, 401, "MODELOPS_UNAUTHORIZED")


def test_modelops_api_lifecycle_audit_and_previous_alias(settings) -> None:
    configured = replace(settings, modelops_token=TOKEN)
    client = TestClient(create_app(configured, registry_with_candidate()))
    initial = client.get("/v1/models").json()
    active = next(item for item in initial["models"] if item["state"] == "active")

    validated = client.post(
        "/v1/models/mock-compliance-v1/validate",
        json={"reason": "security validation passed"},
        headers=HEADERS,
    )
    assert validated.status_code == 200, validated.text
    activated = client.post(
        "/v1/models/mock-compliance-v1/activate",
        json={
            "reason": "controlled rollout",
            "expected_active_model_id": active["model_id"],
            "expected_active_fingerprint": active["model_fingerprint"],
        },
        headers=HEADERS,
    )
    assert activated.status_code == 200, activated.text
    assert activated.json()["registry"]["aliases"]["previous"] == active["model_id"]

    stale = client.post(
        "/v1/models/rollback",
        json={
            "reason": "stale rollback",
            "expected_active_model_id": active["model_id"],
        },
        headers=HEADERS,
    )
    assert_error(stale, 409, "MODELOPS_CONFLICT")
    assert client.get("/v1/models").json()["aliases"]["active"] == "mock-compliance-v1"

    rollback = client.post(
        "/v1/models/rollback",
        json={
            "reason": "operator rollback",
            "expected_active_model_id": "mock-compliance-v1",
        },
        headers=HEADERS,
    )
    assert rollback.status_code == 200, rollback.text
    assert rollback.json()["registry"]["aliases"]["active"] == active["model_id"]

    audit = client.get("/v1/models/audit?limit=20", headers=HEADERS)
    assert audit.status_code == 200
    records = audit.json()["audit"]
    assert [item["action"] for item in records][-3:] == [
        "validate",
        "activate",
        "rollback",
    ]
    assert all(item["actor"] != TOKEN for item in records)


def test_modelops_audit_is_emitted_as_structured_json_without_token() -> None:
    record = logging.LogRecord(
        name="mvis",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="modelops_lifecycle_changed",
        args=(),
        exc_info=None,
    )
    record.audit = {
        "action": "activate",
        "actor": "test-operator",
        "model_fingerprint": "a" * 64,
    }
    payload = json.loads(JsonFormatter().format(record))
    assert payload["audit"]["action"] == "activate"
    assert TOKEN not in json.dumps(payload)


@pytest.mark.parametrize(
    "payload",
    [
        image_bytes("JPEG") + b"<script>alert(1)</script>",
        image_bytes("PNG") + b"PK\x03\x04malicious.zip",
    ],
)
def test_rejects_image_polyglots_with_trailing_payload(client, payload: bytes) -> None:
    content_type = "image/png" if payload.startswith(b"\x89PNG") else "image/jpeg"
    assert_error(
        post_image(client, payload=payload, content_type=content_type),
        400,
        "INVALID_IMAGE",
    )


def test_rejects_highly_compressed_pixel_bomb_before_adapter(client_factory) -> None:
    compressed = image_bytes("PNG", size=(1024, 1024))
    assert len(compressed) < 20_000
    client = client_factory(max_image_pixels=100_000)
    assert_error(
        post_image(client, payload=compressed, content_type="image/png"),
        400,
        "INVALID_IMAGE",
    )


def test_upload_filename_path_injection_is_never_used_as_a_path(
    client, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    response = client.post(
        "/v1/analyze",
        data={"query": "inspect"},
        files={
            "image": (
                "../../../../tmp/model-config.json",
                image_bytes(),
                "image/jpeg",
            )
        },
    )
    assert response.status_code == 200, response.text
    assert list(tmp_path.iterdir()) == []


def test_rejects_encoded_and_decoded_base64_over_limit(client_factory) -> None:
    client = client_factory(max_image_bytes=32)
    encoded_over_limit = "A" * 48
    assert_error(
        client.post(
            "/v1/analyze",
            json={
                "image": f"data:image/jpeg;base64,{encoded_over_limit}",
                "query": "inspect",
            },
        ),
        400,
        "INVALID_IMAGE",
    )

    decoded_over_limit = base64.b64encode(b"x" * 33).decode("ascii")
    assert_error(
        client.post(
            "/v1/analyze",
            json={
                "image": f"data:image/jpeg;base64,{decoded_over_limit}",
                "query": "inspect",
            },
        ),
        400,
        "INVALID_IMAGE",
    )


class FirstCallSlowAdapter(MockModelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.cancelled = False

    async def analyze(self, request: AdapterRequest) -> ModelOutput:
        self.calls += 1
        if self.calls == 1:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return ModelOutput.model_validate(await super().analyze(request))


def test_timeout_cancels_adapter_and_recovers_capacity(client_factory) -> None:
    adapter = FirstCallSlowAdapter()
    client = client_factory(adapter, inference_timeout_seconds=0.005)
    assert_error(post_image(client), 504, "INFERENCE_TIMEOUT")
    recovered = post_image(client)
    assert recovered.status_code == 200, recovered.text
    assert adapter.cancelled is True


def test_concurrency_queue_rejects_overflow_then_recovers(settings) -> None:
    async def scenario() -> None:
        configured = replace(
            settings,
            concurrency_limit=1,
            concurrency_wait_seconds=0.005,
            inference_timeout_seconds=1,
        )
        app = create_app(
            configured,
            build_mock_registry(MockModelAdapter(delay_seconds=0.05)),
        )
        transport = httpx.ASGITransport(app=app)
        form = {"query": "inspect"}
        files = {"image": ("safe.jpg", image_bytes(), "image/jpeg")}
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post("/v1/analyze", data=form, files=files))
            await asyncio.sleep(0.01)
            overflow = await client.post("/v1/analyze", data=form, files=files)
            completed = await first
            recovered = await client.post("/v1/analyze", data=form, files=files)

        assert completed.status_code == 200
        assert_error(overflow, 503, "RESOURCE_EXHAUSTED")
        assert recovered.status_code == 200

    asyncio.run(scenario())
