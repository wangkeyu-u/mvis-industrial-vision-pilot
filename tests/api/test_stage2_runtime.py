from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.analyze_service import AnalyzeCommand, AnalyzeService
from src.core.errors import ErrorCode, ServiceError
from src.core.model_registry import (
    AdapterRequest,
    MockModelAdapter,
    ModelRegistration,
    ModelState,
    build_mock_registry,
)
from src.core.registry_factory import RealAdapterUnavailable, build_service_registry
from src.core.schemas import AnalyzeOptions, AnalyzeTask, ModelIdentity, ModelOutput

from .conftest import image_bytes
from .test_analyze import as_data_url, assert_error, post_image


def unavailable_loader(_settings):
    raise RealAdapterUnavailable("mlx_runtime_unavailable")


class FakeRealAdapter(MockModelAdapter):
    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(base="qwen-real-test", adapter="adapter-real-test")


def successful_real_loader(_settings) -> ModelRegistration:
    return ModelRegistration(
        model_id="qwen-real-test",
        adapter=FakeRealAdapter(),
        state=ModelState.ACTIVE,
        source="test-real-loader",
        quantization="4-bit",
    )


def test_auto_mode_falls_back_explicitly_without_pretending_real(settings) -> None:
    registry = build_service_registry(
        replace(settings, model_mode="auto"), real_loader=unavailable_loader
    )
    status = registry.statuses()

    assert registry.ready()
    assert status["runtime"] == {
        "requested_mode": "auto",
        "selected_mode": "mock",
        "degraded": True,
        "fallback_reason": "mlx_runtime_unavailable",
    }
    assert status["aliases"] == {
        "active": "mock-compliance-v0",
        "candidate": "qwen3-vl-2b-instruct-4bit",
    }
    assert registry.active().adapter.identity.base == "mock-vlm-0"


def test_real_mode_never_silently_falls_back(settings) -> None:
    registry = build_service_registry(
        replace(settings, model_mode="real"), real_loader=unavailable_loader
    )
    client = TestClient(create_app(replace(settings, model_mode="real"), registry))

    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["details"]["runtime"] == {
        "requested_mode": "real",
        "selected_mode": "real",
        "degraded": True,
        "fallback_reason": "mlx_runtime_unavailable",
    }
    assert_error(post_image(client), 503, "MODEL_NOT_READY")


def test_real_loader_can_activate_algorithm_bridge_contract(settings) -> None:
    registry = build_service_registry(
        replace(settings, model_mode="real"), real_loader=successful_real_loader
    )
    assert registry.ready()
    assert registry.runtime_status() == {
        "requested_mode": "real",
        "selected_mode": "real",
        "degraded": False,
        "fallback_reason": None,
    }
    assert registry.active().adapter.identity.base == "qwen-real-test"


def test_cors_allows_only_configured_local_ui_origin(settings) -> None:
    configured = replace(settings, cors_allowed_origins=("http://127.0.0.1:8000",))
    client = TestClient(create_app(configured, build_mock_registry()))
    headers = {
        "Origin": "http://127.0.0.1:8000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-request-id",
    }

    allowed = client.options("/v1/analyze", headers=headers)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == headers["Origin"]
    assert "x-request-id" in allowed.headers["access-control-allow-headers"].lower()

    denied = client.options(
        "/v1/analyze",
        headers=headers | {"Origin": "https://untrusted.example"},
    )
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


class CancellationAwareAdapter(MockModelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.cancelled = asyncio.Event()

    async def analyze(self, request: AdapterRequest) -> ModelOutput:
        self.calls += 1
        if self.calls == 1:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        return ModelOutput.model_validate(await super().analyze(request))


def test_disconnect_cancels_mock_and_releases_capacity(settings) -> None:
    async def scenario() -> None:
        adapter = CancellationAwareAdapter()
        service = AnalyzeService(
            replace(
                settings,
                inference_timeout_seconds=1,
                disconnect_poll_seconds=0.001,
            ),
            build_mock_registry(adapter),
        )
        command = AnalyzeCommand(
            image=image_bytes(),
            image_content_type="image/jpeg",
            query="inspect",
            task=AnalyzeTask.INSPECT,
            model="active",
            use_specialist=False,
            options=AnalyzeOptions(),
        )

        async def disconnected() -> bool:
            return True

        with pytest.raises(ServiceError) as captured:
            await service.analyze(
                command, "req_cancel", cancellation_check=disconnected
            )
        assert captured.value.code is ErrorCode.REQUEST_CANCELLED
        assert adapter.cancelled.is_set()

        response, _ = await service.analyze(command, "req_after_cancel")
        assert response.request_id == "req_after_cancel"

    asyncio.run(scenario())


def test_asgi_disconnect_maps_to_499_and_cancels_adapter(settings) -> None:
    async def scenario() -> None:
        adapter = CancellationAwareAdapter()
        configured = replace(
            settings,
            inference_timeout_seconds=1,
            disconnect_poll_seconds=0.001,
        )
        app = create_app(configured, build_mock_registry(adapter))
        body = json.dumps(
            {"image": as_data_url(image_bytes()), "query": "inspect"}
        ).encode()
        inbound = [
            {"type": "http.request", "body": body, "more_body": False},
            {"type": "http.disconnect"},
        ]
        outbound: list[dict] = []

        async def receive() -> dict:
            if inbound:
                return inbound.pop(0)
            await asyncio.sleep(10)
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
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
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8001),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=0.5)

        starts = [item for item in outbound if item["type"] == "http.response.start"]
        assert starts[0]["status"] == 499
        assert adapter.cancelled.is_set()

    asyncio.run(scenario())


def test_concurrent_request_over_capacity_returns_503(settings) -> None:
    async def scenario() -> None:
        configured = replace(
            settings,
            concurrency_limit=1,
            concurrency_wait_seconds=0.005,
            inference_timeout_seconds=1,
        )
        app = create_app(
            configured,
            build_mock_registry(MockModelAdapter(delay_seconds=0.08)),
        )
        transport = httpx.ASGITransport(app=app)
        form = {"query": "inspect"}
        files = {"image": ("a.jpg", image_bytes(), "image/jpeg")}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = asyncio.create_task(
                client.post("/v1/analyze", data=form, files=files)
            )
            await asyncio.sleep(0.02)
            second = await client.post("/v1/analyze", data=form, files=files)
            first_response = await first

        assert first_response.status_code == 200
        assert_error(second, 503, "RESOURCE_EXHAUSTED")

    asyncio.run(scenario())


def test_one_hundred_mock_requests_are_stable_and_traceable(client) -> None:
    request_ids: set[str] = set()
    for _ in range(100):
        response = post_image(client)
        assert response.status_code == 200
        request_ids.add(response.json()["request_id"])
        assert response.json()["model"]["base"] == "mock-vlm-0"

    assert len(request_ids) == 100
