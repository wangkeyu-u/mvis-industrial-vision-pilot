from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.model_registry import build_mock_registry
from src.core.registry_factory import _build_lora_registration
from src.core.schemas import AnalysisMode, ModelProvenance, ObjectSource
from src.core.specialist_runtime import (
    MockSpecialistAdapter,
    build_mock_specialist_registration,
)

from .conftest import image_bytes
from .test_analyze import assert_error, post_image


def _specialist_client(settings, adapter: MockSpecialistAdapter) -> TestClient:
    registry = build_mock_registry()
    registry.register_specialist(build_mock_specialist_registration(adapter))
    return TestClient(create_app(settings, registry), raise_server_exceptions=False)


def test_specialist_only_returns_score_threshold_bbox_and_bounded_heatmap(settings) -> None:
    client = _specialist_client(
        settings,
        MockSpecialistAdapter(heatmap_png=image_bytes("PNG", size=(8, 6))),
    )

    response = post_image(client, analysis_mode="specialist_only")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["analysis_mode"] == "specialist_only"
    assert body["result"] == "violation"
    assert body["specialist"]["score"] == 0.9
    assert body["specialist"]["threshold"] == 0.7
    assert body["specialist"]["source"] == "patchcore"
    assert body["objects"] == body["specialist"]["objects"]
    assert body["objects"][0]["source"] == "patchcore"
    artifact = body["specialist"]["heatmap"]
    assert set(artifact) == {
        "artifact_id",
        "uri",
        "sha256",
        "media_type",
        "width",
        "height",
        "expires_in_seconds",
    }
    fetched = client.get(artifact["uri"])
    assert fetched.status_code == 200
    assert fetched.headers["content-type"] == "image/png"
    assert fetched.headers["cache-control"] == "private, max-age=0, no-store"
    assert fetched.content == image_bytes("PNG", size=(8, 6))


def test_fused_conflict_keeps_only_specialist_localization(settings) -> None:
    client = _specialist_client(settings, MockSpecialistAdapter())

    response = post_image(client, analysis_mode="fused")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["analysis_mode"] == "fused"
    assert body["result"] == "uncertain"
    assert body["uncertain"] is True
    assert body["human_review_required"] is True
    assert body["objects"] == body["specialist"]["objects"]
    assert all(item["source"] == "patchcore" for item in body["objects"])
    assert [8.0, 6.0, 24.0, 18.0] not in [item["bbox"] for item in body["objects"]]
    assert "model_conflict" in body["warnings"]
    assert "specialist_localization_authoritative" in body["warnings"]


def test_fused_vlm_positive_specialist_negative_discards_vlm_boxes(settings) -> None:
    client = _specialist_client(
        settings,
        MockSpecialistAdapter(score=0.2, threshold=0.7),
    )

    response = post_image(client, analysis_mode="fused")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"] == "uncertain"
    assert body["objects"] == []
    assert body["specialist"]["detected"] is False
    assert body["human_review_required"] is True
    assert "vlm_localization_discarded" in body["warnings"]


def test_fused_agreement_still_replaces_vlm_source_with_specialist(settings) -> None:
    adapter = MockSpecialistAdapter(bbox=(8.0, 6.0, 24.0, 18.0))
    client = _specialist_client(settings, adapter)

    response = post_image(client, analysis_mode="fused")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"] == "violation"
    assert body["uncertain"] is False
    assert body["human_review_required"] is False
    assert {item["source"] for item in body["objects"]} == {ObjectSource.PATCHCORE}


def test_specialist_mode_is_explicitly_not_ready_without_artifact(settings) -> None:
    configured = replace(settings, default_analysis_mode=AnalysisMode.SPECIALIST_ONLY.value)
    client = TestClient(create_app(configured, build_mock_registry()))

    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["details"]["analysis_modes"]["specialist_only"][
        "runtime_ready"
    ] is False
    assert_error(post_image(client, analysis_mode="specialist_only"), 404, "MODEL_NOT_FOUND")


def test_invalid_mode_and_heatmap_artifact_ids_have_stable_errors(settings) -> None:
    client = _specialist_client(settings, MockSpecialistAdapter())
    assert_error(post_image(client, analysis_mode="invented"), 400, "INVALID_QUERY")
    assert_error(
        client.get("/v1/artifacts/heatmaps/hm_not-a-valid-id"),
        404,
        "ARTIFACT_NOT_FOUND",
    )
    assert_error(client.get("/v1/artifacts/heatmaps/hm_" + "0" * 32), 404, "ARTIFACT_NOT_FOUND")


def test_invalid_or_oversized_heatmap_is_rejected_at_output_boundary(settings) -> None:
    invalid = _specialist_client(
        settings,
        MockSpecialistAdapter(heatmap_png=b"not-a-png"),
    )
    assert_error(
        post_image(invalid, analysis_mode="specialist_only"),
        422,
        "OUTPUT_VALIDATION_FAILED",
    )

    oversized = _specialist_client(
        replace(settings, heatmap_max_pixels=4),
        MockSpecialistAdapter(heatmap_png=image_bytes("PNG", size=(3, 2))),
    )
    assert_error(
        post_image(oversized, analysis_mode="specialist_only"),
        422,
        "OUTPUT_VALIDATION_FAILED",
    )


class FirstSpecialistCallSlow(MockSpecialistAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.cancelled = False

    async def detect(self, request):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super().detect(request)


def test_fused_timeout_cancels_both_branch_and_recovers_capacity(settings) -> None:
    adapter = FirstSpecialistCallSlow()
    client = _specialist_client(
        replace(settings, inference_timeout_seconds=0.005),
        adapter,
    )

    assert_error(post_image(client, analysis_mode="fused"), 504, "INFERENCE_TIMEOUT")
    recovered = post_image(client, analysis_mode="fused")

    assert recovered.status_code == 200, recovered.text
    assert adapter.cancelled is True


def test_thirty_fused_mock_requests_are_stable_and_traceable(settings) -> None:
    async def scenario() -> None:
        registry = build_mock_registry()
        registry.register_specialist(build_mock_specialist_registration())
        app = create_app(settings, registry)
        transport = httpx.ASGITransport(app=app)
        ids: set[str] = set()
        files = {"image": ("safe.jpg", image_bytes(), "image/jpeg")}
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(30):
                response = await client.post(
                    "/v1/analyze",
                    data={"query": "inspect", "analysis_mode": "fused"},
                    files=files,
                )
                assert response.status_code == 200, response.text
                ids.add(response.headers["x-request-id"])
                assert response.json()["analysis_mode"] == "fused"
        assert len(ids) == 30

    asyncio.run(scenario())


def test_mode_quality_gate_requires_both_signed_components(settings) -> None:
    registry = build_mock_registry()
    registry.register_specialist(build_mock_specialist_registration())
    client = TestClient(create_app(settings, registry))

    statuses = client.get("/v1/models").json()["analysis_modes"]

    assert statuses["vlm_only"]["quality_accepted"] is False
    assert statuses["specialist_only"]["quality_accepted"] is False
    assert statuses["fused"]["quality_accepted"] is False
    assert statuses["fused"]["serving_tier"] == "pilot"
    assert statuses["fused"]["production_ready"] is False
    version = client.get("/version").json()
    assert version["runtime"]["default_analysis_mode"] == "vlm_only"
    assert version["runtime"]["specialist"]["provenance"][
        "checkpoint_revision"
    ] == "mock-specialist-v0"


def test_lora_run_manifest_hash_mismatch_keeps_explicit_candidate_unready(
    settings, tmp_path
) -> None:
    adapter = tmp_path / "adapters.safetensors"
    adapter.write_bytes(b"bounded-adapter-fixture")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_kind": "formal",
                "artifacts": {"adapter_sha256": "0" * 64},
                "model": {
                    "revision": "revision-a",
                    "weight_sha256": "1" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    configured = replace(
        settings,
        lora_adapter_path=str(adapter),
        lora_run_manifest_path=str(manifest),
    )
    base = ModelProvenance(
        model_id="base-model",
        checkpoint_revision="revision-a",
        backend="mlx_vlm",
        config_fingerprint="2" * 64,
        config_artifact_sha256="3" * 64,
        weight_hash="1" * 64,
        data_version="data-v1",
        prompt_version="prompt-v1",
    )

    registration = _build_lora_registration(configured, base)

    assert registration.adapter.ready is False
    assert registration.adapter.reason_code == "lora_adapter_hash_mismatch"
    assert registration.quality_accepted is False
