from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace

import pytest

from src.core.model_registry import (
    MockModelAdapter,
    ModelRegistration,
    ModelRegistry,
    ModelState,
    build_mock_registry,
    fingerprint_mapping,
)
from src.core.quality_gate import (
    EvaluatorPayload,
    canonical_payload,
    verify_evaluator_attestation,
)
from src.core.registry_factory import (
    _artifact_fingerprint,
    _configured_provenance,
    _unavailable_registry,
)
from src.core.schemas import (
    ModelProvenance,
    QualityEvidence,
    QualityStatus,
    ServingTier,
)
from src.observability.candidate_validation import run_lora_candidate_probe

from .test_analyze import post_image


def provenance(*, adapter_hash: str | None = None) -> ModelProvenance:
    return ModelProvenance(
        model_id="mlx-community/Qwen3-VL-2B-Instruct-4bit",
        checkpoint_revision="9c4f5209e57b31f4b9dfba735de3fb983739c9cc",
        backend="mlx_vlm",
        config_fingerprint="a" * 64,
        adapter_hash=adapter_hash,
        weight_hash="b" * 64,
        data_version="ksdd-0.1.0",
        prompt_version="prompt_v1",
    )


def write_attestation(tmp_path, model_provenance: ModelProvenance, key: str):
    payload = EvaluatorPayload(
        model_id=model_provenance.model_id,
        checkpoint_revision=model_provenance.checkpoint_revision,
        config_fingerprint=model_provenance.config_fingerprint,
        adapter_hash=model_provenance.adapter_hash,
        weight_hash=model_provenance.weight_hash,
        data_version=model_provenance.data_version,
        prompt_version=model_provenance.prompt_version,
        quality_status="pilot_passed",
        eligible_for_model_acceptance=True,
        evaluator="mvis-offline-evaluator-v1",
        issued_at="2026-08-09T00:00:00Z",
        report_sha256="c" * 64,
    )
    signature = hmac.new(
        key.encode(),
        canonical_payload(payload),
        hashlib.sha256,
    ).hexdigest()
    path = tmp_path / "evaluator-attestation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "payload": payload.model_dump(mode="json"),
                "signature": {
                    "algorithm": "hmac-sha256",
                    "key_id": "evaluation-ci-v1",
                    "value": signature,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_revision_and_provenance_cross_all_public_api_surfaces(client) -> None:
    analyze = post_image(client).json()
    version = client.get("/version").json()
    models = client.get("/v1/models").json()
    active = next(item for item in models["models"] if item["state"] == "active")

    for payload in (analyze["provenance"], version["provenance"], active["provenance"]):
        assert payload["checkpoint_revision"] == "mock-builtin-v0"
        assert payload["data_version"] == "synthetic-contract-v0"
        assert payload["prompt_version"] == "mock-contract-v0"
        assert "not_reported" not in json.dumps(payload)
    assert analyze["model"]["revision"] == version["model"]["revision"]
    assert version["quality_accepted"] is False
    assert version["runtime"]["runtime_ready"] is True
    assert version["runtime"]["production_ready"] is False


def test_real_config_provenance_matches_algorithm_contract(settings) -> None:
    configured = replace(
        settings,
        model_config_path="configs/models/qwen3_vl_2b_mlx_4bit.json",
        model_weight_hash="b" * 64,
        data_version="ksdd-0.1.0",
        prompt_version="prompt_v1",
    )
    actual = _configured_provenance(configured)

    assert actual.model_id == "mlx-community/Qwen3-VL-2B-Instruct-4bit"
    assert actual.checkpoint_revision == "9c4f5209e57b31f4b9dfba735de3fb983739c9cc"
    assert actual.config_fingerprint == "71b87d18c01aeb4d"
    assert len(actual.config_artifact_sha256) == 64
    assert actual.data_version == "ksdd-0.1.0"
    assert actual.prompt_version == "prompt_v1"


def test_evaluator_attestation_binds_signature_to_exact_provenance(tmp_path) -> None:
    model_provenance = provenance(adapter_hash="d" * 64)
    key = "phase6-evaluator-secret-key-32bytes-minimum"
    path = write_attestation(tmp_path, model_provenance, key)

    status, evidence = verify_evaluator_attestation(path, key, model_provenance)
    assert status is QualityStatus.PILOT_PASSED
    assert evidence.signature_verified is True
    assert evidence.report_sha256 == "c" * 64

    mismatched = model_provenance.model_copy(update={"prompt_version": "prompt_v2"})
    status, evidence = verify_evaluator_attestation(path, key, mismatched)
    assert status is QualityStatus.UNVALIDATED
    assert evidence.signature_verified is False
    assert evidence.failure_reason == "provenance_mismatch"


def test_adapter_hash_matches_weights_and_rejects_symlinked_tree(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapters.safetensors"
    weights.write_bytes(b"controlled-lora-weights")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert _artifact_fingerprint(adapter) == hashlib.sha256(b"controlled-lora-weights").hexdigest()

    (adapter / "unsafe-link").symlink_to(tmp_path / "outside")
    assert _artifact_fingerprint(adapter) is None


def test_production_activation_requires_signed_quality_but_pilot_is_allowed() -> None:
    registry = build_mock_registry()
    candidate = ModelRegistration(
        model_id="candidate",
        adapter=MockModelAdapter(base="candidate", adapter_id="lora-r8"),
        state=ModelState.CANDIDATE,
        source="test-candidate",
        config_fingerprint=fingerprint_mapping({"candidate": 1}),
        provenance=provenance(adapter_hash="d" * 64),
        quality_status=QualityStatus.PILOT_CANDIDATE,
    )
    registry.register(candidate)
    registry.validate_candidate(
        "candidate",
        actor="test",
        request_id="req_validate",
        reason="runtime probe passed",
    )
    before = registry.statuses()
    with pytest.raises(ValueError, match="signed evaluator report"):
        registry.activate(
            "candidate",
            actor="test",
            request_id="req_production",
            reason="unsafe production attempt",
            serving_tier=ServingTier.PRODUCTION,
        )
    assert registry.statuses() == before

    registry.activate(
        "candidate",
        actor="test",
        request_id="req_pilot",
        reason="explicit pilot rollout",
        serving_tier=ServingTier.PILOT,
    )
    assert registry.ready() is True
    assert registry.production_ready() is False
    assert registry.active().serving_tier is ServingTier.PILOT


def test_verified_candidate_can_atomically_activate_and_rollback_production() -> None:
    registry = ModelRegistry()
    evidence = QualityEvidence(
        signature_verified=True,
        algorithm="hmac-sha256",
        key_id="evaluation-ci-v1",
        report_sha256="c" * 64,
        attestation_sha256="e" * 64,
    )
    for model_id, state in (("zero-shot", ModelState.ACTIVE), ("lora", ModelState.CANDIDATE)):
        registry.register(
            ModelRegistration(
                model_id=model_id,
                adapter=MockModelAdapter(base=model_id, adapter_id=model_id),
                state=state,
                source="test",
                provenance=provenance(adapter_hash=("d" * 64 if model_id == "lora" else None)),
                quality_status=QualityStatus.PILOT_PASSED,
                quality_evidence=evidence,
                serving_tier=(ServingTier.PRODUCTION if state is ModelState.ACTIVE else None),
            )
        )
    registry.validate_candidate(
        "lora", actor="test", request_id="req_validate", reason="signed report matched"
    )
    registry.activate(
        "lora",
        actor="test",
        request_id="req_activate",
        reason="production switch",
        serving_tier=ServingTier.PRODUCTION,
    )
    assert registry.production_ready() is True
    assert registry.statuses()["aliases"]["previous"] == "zero-shot"

    registry.rollback(
        actor="test",
        request_id="req_rollback",
        reason="atomic rollback",
        serving_tier=ServingTier.PRODUCTION,
    )
    assert registry.active().model_id == "zero-shot"
    assert registry.production_ready() is True


def test_real_unavailable_registry_still_exposes_zero_shot_and_lora_entries(settings) -> None:
    configured = replace(
        settings,
        model_mode="real",
        model_weight_hash="b" * 64,
        lora_adapter_path=None,
    )
    registry = _unavailable_registry(
        configured,
        "model_load_failed",
        include_lora=True,
    )
    statuses = registry.statuses()
    by_id = {item["model_id"]: item for item in statuses["models"]}

    assert statuses["aliases"]["zero_shot"] == "qwen3-vl-2b-instruct-4bit"
    assert statuses["aliases"]["lora"] == configured.lora_model_id
    assert by_id["qwen3-vl-2b-instruct-4bit"]["quality_status"] == "pilot_failed"
    assert by_id[configured.lora_model_id]["quality_status"] == "unvalidated"
    assert by_id[configured.lora_model_id]["runtime_ready"] is False
    assert by_id[configured.lora_model_id]["provenance"]["adapter_hash"] is None
    assert by_id[configured.lora_model_id]["provenance"]["data_version"] == "ksdd_sft-1.0.0"
    assert by_id[configured.lora_model_id]["provenance"]["prompt_version"] == "ksdd_prompt_v1"


def test_candidate_probe_blocks_without_artifact_and_never_uses_mock(settings) -> None:
    report = run_lora_candidate_probe(replace(settings, model_mode="real", lora_adapter_path=None))
    assert report["status"] == "blocked"
    assert report["mock_used"] is False
    assert report["downloads_allowed"] is False
    assert report["blockers"] == ["lora_artifact_unavailable"]
    assert report["candidate"]["quality_status"] == "unvalidated"


def test_candidate_probe_includes_bounded_training_resource_evidence(settings, tmp_path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "resource_limit_exceeded",
                "run_kind": "smoke",
                "error": {"type": "MemoryLimitExceeded", "message": "peak exceeded"},
                "resources": {
                    "mlx_peak_allocated_mb": 15997.066,
                    "process_peak_rss_mb": 1869.938,
                    "memory_limit_mb": 12288,
                },
                "artifacts": {"adapter_path": None, "adapter_sha256": None},
            }
        ),
        encoding="utf-8",
    )
    report = run_lora_candidate_probe(
        replace(settings, model_mode="real", lora_adapter_path=None),
        run_manifest_path=manifest,
    )
    assert report["blockers"] == [
        "lora_artifact_unavailable",
        "training_resource_limit_exceeded",
    ]
    assert report["algorithm_diagnostics"]["mlx_peak_allocated_mb"] == 15997.066
    assert report["algorithm_diagnostics"]["adapter_available"] is False
    assert len(report["algorithm_diagnostics"]["manifest_sha256"]) == 64
