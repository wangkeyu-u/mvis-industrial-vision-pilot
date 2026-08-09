"""Cryptographic evaluator-attestation verification for production activation.

The evaluator signing key is environment-only.  A report may describe pilot
results without a signature, but only a matching, verified attestation can make
a model production-eligible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.schemas import ModelProvenance, QualityEvidence, QualityStatus

MAX_ATTESTATION_BYTES = 1024 * 1024


class EvaluatorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    checkpoint_revision: str
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{16,64}$")
    adapter_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    weight_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    data_version: str
    prompt_version: str
    quality_status: Literal["pilot_passed"]
    eligible_for_model_acceptance: Literal[True]
    evaluator: str
    issued_at: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluatorSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["hmac-sha256"]
    key_id: str
    value: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluatorAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"]
    payload: EvaluatorPayload
    signature: EvaluatorSignature


def canonical_payload(payload: EvaluatorPayload) -> bytes:
    return json.dumps(
        payload.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def verify_evaluator_attestation(
    path: str | Path | None,
    signing_key: str | None,
    provenance: ModelProvenance,
) -> tuple[QualityStatus, QualityEvidence]:
    """Verify a bounded HMAC attestation and bind it to exact model provenance."""

    if path is None:
        return QualityStatus.UNVALIDATED, QualityEvidence(failure_reason="attestation_missing")
    if signing_key is None:
        return QualityStatus.UNVALIDATED, QualityEvidence(failure_reason="signing_key_missing")
    if len(signing_key.encode("utf-8")) < 32:
        return QualityStatus.UNVALIDATED, QualityEvidence(failure_reason="signing_key_too_short")

    candidate = Path(path)
    try:
        stat = candidate.stat()
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or stat.st_size <= 0
            or stat.st_size > MAX_ATTESTATION_BYTES
        ):
            return QualityStatus.UNVALIDATED, QualityEvidence(
                failure_reason="attestation_file_invalid"
            )
        raw = candidate.read_bytes()
        attestation = EvaluatorAttestation.model_validate_json(raw)
    except (OSError, ValidationError):
        return QualityStatus.UNVALIDATED, QualityEvidence(failure_reason="attestation_parse_failed")

    payload = attestation.payload
    expected = hmac.new(
        signing_key.encode("utf-8"),
        canonical_payload(payload),
        hashlib.sha256,
    ).hexdigest()
    evidence_base = {
        "algorithm": attestation.signature.algorithm,
        "key_id": attestation.signature.key_id,
        "report_sha256": payload.report_sha256,
        "attestation_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if not hmac.compare_digest(expected, attestation.signature.value):
        return QualityStatus.UNVALIDATED, QualityEvidence(
            **evidence_base,
            failure_reason="signature_invalid",
        )

    expected_identity = (
        provenance.model_id,
        provenance.checkpoint_revision,
        provenance.config_fingerprint,
        provenance.adapter_hash,
        provenance.weight_hash,
        provenance.data_version,
        provenance.prompt_version,
    )
    attested_identity = (
        payload.model_id,
        payload.checkpoint_revision,
        payload.config_fingerprint,
        payload.adapter_hash,
        payload.weight_hash,
        payload.data_version,
        payload.prompt_version,
    )
    if not hmac.compare_digest(
        json.dumps(expected_identity, separators=(",", ":")),
        json.dumps(attested_identity, separators=(",", ":")),
    ):
        return QualityStatus.UNVALIDATED, QualityEvidence(
            **evidence_base,
            failure_reason="provenance_mismatch",
        )

    return QualityStatus.PILOT_PASSED, QualityEvidence(
        signature_verified=True,
        **evidence_base,
    )
