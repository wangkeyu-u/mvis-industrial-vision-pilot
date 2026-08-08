"""Live-network tests for FastAPI -> same-origin proxy -> HTTP client contract.

This file is intentionally outside unittest's ``test_*.py`` pattern because
it requires the FastAPI test dependencies. Run it with the documented uv
command in docs/qa.
"""

from __future__ import annotations

import base64
import io
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from PIL import Image

from app.server import create_server
from src.api.app import create_app
from src.core.model_registry import AdapterRequest, MockModelAdapter, build_mock_registry
from src.core.schemas import EvidenceObject, ModelIdentity, ModelOutput, ObjectSource


class PolicyStateAdapter:
    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(base="e2e-policy-adapter", adapter="e2e-v1")

    @property
    def ready(self) -> bool:
        return True

    async def analyze(self, request: AdapterRequest) -> ModelOutput:
        query = request.query.casefold()
        if "refuse" in query or "拒答" in query:
            return ModelOutput(
                result="uncertain",
                objects=[],
                reason="Evidence is insufficient; automatic judgment refused.",
                uncertain=True,
                warnings=["refusal:insufficient_evidence"],
            )
        if "uncertain" in query or "不确定" in query:
            return ModelOutput(
                result="uncertain",
                objects=[],
                reason="Evidence confidence is below the review threshold.",
                uncertain=True,
                warnings=["low_confidence"],
            )
        return ModelOutput(
            result="violation",
            objects=[
                EvidenceObject(
                    label="e2e_target",
                    bbox=(4, 3, 20, 15),
                    confidence=0.88,
                    source=ObjectSource.MOCK,
                )
            ],
            reason="Live FastAPI e2e evidence.",
            uncertain=False,
            warnings=["e2e_adapter"],
        )


class LiveUvicorn:
    def __init__(self, application) -> None:  # type: ignore[no-untyped-def]
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(128)
        self.host, self.port = self.socket.getsockname()
        self.server = uvicorn.Server(
            uvicorn.Config(application, host=self.host, port=self.port, log_level="warning", lifespan="off")
        )
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 5
        while not self.server.started and time.time() < deadline:
            time.sleep(0.01)
        if not self.server.started:
            raise RuntimeError("uvicorn did not start")

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self.socket.close()


class LiveProxy:
    def __init__(self, backend_url: str) -> None:
        self.server = create_server(port=0, backend_url=backend_url, proxy_timeout=2)
        self.host, self.port = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (210, 210, 210)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def live_services():  # type: ignore[no-untyped-def]
    ready_backend = LiveUvicorn(create_app(registry=build_mock_registry(PolicyStateAdapter())))
    not_ready_backend = LiveUvicorn(create_app(registry=build_mock_registry(MockModelAdapter(ready=False))))
    ready_backend.start()
    not_ready_backend.start()
    ready_proxy = LiveProxy(f"http://{ready_backend.host}:{ready_backend.port}")
    not_ready_proxy = LiveProxy(f"http://{not_ready_backend.host}:{not_ready_backend.port}")
    ready_proxy.start()
    not_ready_proxy.start()
    try:
        yield ready_proxy, not_ready_proxy
    finally:
        ready_proxy.close()
        not_ready_proxy.close()
        ready_backend.close()
        not_ready_backend.close()


def post_multipart(client: httpx.Client, query: str, request_id: str):  # type: ignore[no-untyped-def]
    return client.post(
        "/v1/analyze",
        headers={"X-Request-ID": request_id},
        data={
            "query": query,
            "task": "inspect",
            "model": "active",
            "use_specialist": "true",
            "options": '{"temperature":0,"seed":42}',
        },
        files={"image": ("case.png", png_bytes(), "image/png")},
    )


@pytest.mark.parametrize(
    ("query", "result", "warning"),
    [
        ("find violation", "violation", "e2e_adapter"),
        ("uncertain evidence", "uncertain", "low_confidence"),
        ("refuse insufficient evidence", "uncertain", "refusal:insufficient_evidence"),
    ],
)
def test_live_multipart_success_uncertain_and_refusal(live_services, query: str, result: str, warning: str) -> None:  # type: ignore[no-untyped-def]
    ready_proxy, _ = live_services
    request_id = f"req_e2e_{result}_{warning.split(':')[0]}"
    with httpx.Client(base_url=f"http://{ready_proxy.host}:{ready_proxy.port}", timeout=3) as client:
        response = post_multipart(client, query, request_id)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["request_id"] == request_id == response.headers["x-request-id"]
    assert payload["result"] == result
    assert warning in payload["warnings"]
    assert response.headers["x-mvis-backend"] == "fastapi-proxy"
    if result == "uncertain":
        assert payload["objects"] == []
        assert payload["uncertain"] is True
    else:
        assert payload["objects"][0]["bbox"] == [4.0, 3.0, 20.0, 15.0]


def test_live_json_transport_and_invalid_multipart_error(live_services) -> None:  # type: ignore[no-untyped-def]
    ready_proxy, _ = live_services
    encoded = base64.b64encode(png_bytes()).decode("ascii")
    with httpx.Client(base_url=f"http://{ready_proxy.host}:{ready_proxy.port}", timeout=3) as client:
        json_response = client.post(
            "/v1/analyze",
            headers={"X-Request-ID": "req_e2e_json"},
            json={
                "image": f"data:image/png;base64,{encoded}",
                "query": "find violation",
                "task": "inspect",
                "model": "active",
                "use_specialist": False,
            },
        )
        invalid = client.post(
            "/v1/analyze",
            headers={"X-Request-ID": "req_e2e_invalid"},
            data={"query": "inspect"},
            files={"image": ("fake.png", b"plain text", "image/png")},
        )
    assert json_response.status_code == 200
    assert json_response.json()["request_id"] == "req_e2e_json"
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "INVALID_IMAGE"
    assert invalid.json()["request_id"] == "req_e2e_invalid" == invalid.headers["x-request-id"]


def test_live_model_not_ready_passes_structured_error(live_services) -> None:  # type: ignore[no-untyped-def]
    _, not_ready_proxy = live_services
    with httpx.Client(base_url=f"http://{not_ready_proxy.host}:{not_ready_proxy.port}", timeout=3) as client:
        ready = client.get("/health/ready", headers={"X-Request-ID": "req_e2e_ready"})
        response = post_multipart(client, "inspect", "req_e2e_not_ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "MODEL_NOT_READY"
    assert payload["request_id"] == "req_e2e_not_ready" == response.headers["x-request-id"]
