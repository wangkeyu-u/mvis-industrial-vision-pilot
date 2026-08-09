from __future__ import annotations

import base64
import json

import pytest

from src.core.model_registry import MockModelAdapter

from .conftest import image_bytes


def post_image(client, *, payload: bytes | None = None, content_type: str = "image/jpeg", **data):
    form = {"query": "找出不符合要求的区域", **data}
    return client.post(
        "/v1/analyze",
        data=form,
        files={"image": ("untrusted-name.jpg", payload or image_bytes(), content_type)},
    )


def assert_error(response, status: int, code: str) -> None:
    assert response.status_code == status
    body = response.json()["error"]
    assert body["code"] == code
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["message"]


def test_analyze_returns_versioned_schema_and_original_pixel_bbox(client) -> None:
    response = post_image(
        client,
        task="inspect",
        use_specialist="true",
        options=json.dumps({"temperature": 0, "seed": 42}),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "1.0.0"
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["model"] == {
        "base": "mock-vlm-0",
        "adapter": "mock-compliance-v0",
        "revision": "mock-builtin-v0",
    }
    assert body["provenance"]["checkpoint_revision"] == "mock-builtin-v0"
    assert body["quality_status"] == "unvalidated"
    assert body["quality_accepted"] is False
    assert body["serving_tier"] == "pilot"
    assert body["result"] == "violation"
    assert body["objects"][0]["bbox"] == [8.0, 6.0, 24.0, 18.0]
    assert body["objects"][0]["source"] == "mock"
    assert body["uncertain"] is True
    assert set(body["latency"]) == {
        "preprocess_ms",
        "inference_ms",
        "validation_ms",
    }
    assert body["timing"] == body["latency"]


def as_data_url(payload: bytes, content_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def test_json_base64_contract_used_by_http_client(client) -> None:
    response = client.post(
        "/v1/analyze",
        json={
            "image": as_data_url(image_bytes()),
            "image_width": 32,
            "image_height": 24,
            "query": "找出不合规的区域",
            "task": "inspect",
            "model": "active",
            "use_specialist": True,
            "options": {"temperature": 0, "seed": 42},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["objects"][0]["bbox"] == [8.0, 6.0, 24.0, 18.0]


@pytest.mark.parametrize(
    ("image", "query", "code"),
    [
        ("data:image/jpeg;base64,not-valid@@", "inspect", "INVALID_IMAGE"),
        (as_data_url(image_bytes("PNG"), "image/jpeg"), "inspect", "INVALID_IMAGE"),
        (as_data_url(image_bytes()), " ", "INVALID_QUERY"),
    ],
)
def test_json_transport_rejects_invalid_image_or_query(
    client, image: str, query: str, code: str
) -> None:
    assert_error(client.post("/v1/analyze", json={"image": image, "query": query}), 400, code)


def test_json_transport_maps_malformed_json_to_stable_error(client) -> None:
    response = client.post(
        "/v1/analyze",
        content=b'{"image":',
        headers={"Content-Type": "application/json"},
    )
    assert_error(response, 400, "INVALID_QUERY")


def test_openapi_documents_multipart_and_json_transports(client) -> None:
    content = client.get("/openapi.json").json()["paths"]["/v1/analyze"]["post"]["requestBody"][
        "content"
    ]
    assert "multipart/form-data" in content
    assert "application/json" in content


@pytest.mark.parametrize(
    ("image_format", "mime"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_supported_image_formats(client, image_format: str, mime: str) -> None:
    response = post_image(client, payload=image_bytes(image_format), content_type=mime)
    assert response.status_code == 200, response.text


def test_negative_conclusion_has_no_fabricated_evidence(client) -> None:
    response = client.post(
        "/v1/analyze",
        data={"query": "未发现不合规项"},
        files={"image": ("a.jpg", image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["result"] == "compliant"
    assert response.json()["objects"] == []


def test_mock_does_not_misread_noncompliant_as_compliant(client) -> None:
    response = client.post(
        "/v1/analyze",
        data={"query": "找出不合规的区域"},
        files={"image": ("a.jpg", image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["result"] == "violation"


@pytest.mark.parametrize(
    ("payload", "mime"),
    [
        (image_bytes("PNG"), "image/jpeg"),
        (b"\xff\xd8\xffnot-a-decodable-image", "image/jpeg"),
        (b"plain text", "text/plain"),
    ],
)
def test_rejects_mime_spoof_and_corrupt_content(client, payload: bytes, mime: str) -> None:
    assert_error(post_image(client, payload=payload, content_type=mime), 400, "INVALID_IMAGE")


def test_rejects_byte_limit_before_decode(client_factory) -> None:
    client = client_factory(max_image_bytes=32)
    assert_error(post_image(client, payload=image_bytes()), 400, "INVALID_IMAGE")


def test_rejects_decoded_pixel_limit(client_factory) -> None:
    client = client_factory(max_image_pixels=100)
    assert_error(
        post_image(client, payload=image_bytes(size=(20, 20))),
        400,
        "INVALID_IMAGE",
    )


@pytest.mark.parametrize("query", ["", " ", "x" * 1001])
def test_rejects_invalid_query(client, query: str) -> None:
    response = client.post(
        "/v1/analyze",
        data={"query": query},
        files={"image": ("a.jpg", image_bytes(), "image/jpeg")},
    )
    assert_error(response, 400, "INVALID_QUERY")


def test_rejects_unknown_options_and_task(client) -> None:
    assert_error(post_image(client, options='{"unknown": true}'), 400, "INVALID_QUERY")
    assert_error(post_image(client, task="delete_everything"), 400, "INVALID_QUERY")


def test_model_not_found(client) -> None:
    assert_error(post_image(client, model="missing"), 404, "MODEL_NOT_FOUND")


def test_model_not_ready(client_factory) -> None:
    client = client_factory(MockModelAdapter(ready=False))
    assert_error(post_image(client), 503, "MODEL_NOT_READY")


def test_inference_timeout(client_factory) -> None:
    client = client_factory(MockModelAdapter(delay_seconds=0.05), inference_timeout_seconds=0.005)
    assert_error(post_image(client), 504, "INFERENCE_TIMEOUT")


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        ("invalid_output", 422, "OUTPUT_VALIDATION_FAILED"),
        ("memory", 503, "RESOURCE_EXHAUSTED"),
        ("internal", 500, "INTERNAL_ERROR"),
    ],
)
def test_adapter_failures_map_to_stable_errors(
    client_factory, failure: str, status: int, code: str
) -> None:
    client = client_factory(MockModelAdapter(failure=failure))
    assert_error(post_image(client), status, code)


def test_request_id_is_accepted_only_when_safe(client) -> None:
    response = client.post(
        "/v1/analyze",
        headers={"X-Request-ID": "client-safe_123"},
        data={"query": "inspect"},
        files={"image": ("a.jpg", image_bytes(), "image/jpeg")},
    )
    assert response.headers["x-request-id"] == "client-safe_123"

    rejected = client.get("/health/live", headers={"X-Request-ID": "bad id\nvalue"})
    assert rejected.headers["x-request-id"].startswith("req_")
