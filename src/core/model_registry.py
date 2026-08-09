"""ModelAdapter contract and a small lifecycle-aware local registry."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from src.core.errors import ErrorCode, ServiceError
from src.core.schemas import (
    AnalysisMode,
    AnalyzeOptions,
    AnalyzeTask,
    EvidenceObject,
    ImageMetadata,
    ModelIdentity,
    ModelOutput,
    ModelProvenance,
    ObjectSource,
    QualityEvidence,
    QualityStatus,
    ServingTier,
)
from src.core.specialist_runtime import SpecialistRegistration


class ModelState(StrEnum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    RETIRED = "retired"


ALLOWED_TRANSITIONS: dict[ModelState, set[ModelState]] = {
    ModelState.CANDIDATE: {ModelState.VALIDATED, ModelState.RETIRED},
    ModelState.VALIDATED: {ModelState.ACTIVE, ModelState.RETIRED},
    ModelState.ACTIVE: {ModelState.VALIDATED, ModelState.RETIRED},
    ModelState.RETIRED: set(),
}


def fingerprint_mapping(value: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 for non-secret model metadata."""

    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class AdapterRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_bytes: bytes
    image: ImageMetadata
    request_id: str = ""
    query: str
    task: AnalyzeTask
    use_specialist: bool
    options: AnalyzeOptions


@runtime_checkable
class ModelAdapter(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    @property
    def ready(self) -> bool: ...

    async def analyze(self, request: AdapterRequest) -> ModelOutput | Mapping: ...


@dataclass(slots=True)
class ModelRegistration:
    model_id: str
    adapter: ModelAdapter
    state: ModelState
    source: str
    quantization: str | None = None
    weight_hash: str | None = None
    config_fingerprint: str | None = None
    model_fingerprint: str | None = None
    provenance: ModelProvenance | None = None
    quality_status: QualityStatus = QualityStatus.UNVALIDATED
    quality_evidence: QualityEvidence | None = None
    serving_tier: ServingTier | None = None

    def __post_init__(self) -> None:
        if self.config_fingerprint is None and not self.source.startswith("model-config:"):
            self.config_fingerprint = fingerprint_mapping(
                {
                    "model_id": self.model_id,
                    "source": self.source,
                    "quantization": self.quantization,
                }
            )
        if self.provenance is None:
            identity = self.adapter.identity
            self.provenance = ModelProvenance(
                model_id=identity.base,
                checkpoint_revision=identity.revision,
                backend="mock" if self.source == "built-in-test-double" else "unreported-backend",
                config_fingerprint=self.config_fingerprint or "0" * 64,
                config_artifact_sha256=self.config_fingerprint,
                adapter_hash=None,
                weight_hash=self.weight_hash,
                data_version="synthetic-contract-v0",
                prompt_version="mock-contract-v0",
            )
        if self.quality_evidence is None:
            self.quality_evidence = QualityEvidence(failure_reason="attestation_missing")
        if self.state is ModelState.ACTIVE and self.serving_tier is None:
            self.serving_tier = ServingTier.PILOT
        if self.model_fingerprint is None:
            self.model_fingerprint = fingerprint_mapping(
                {
                    "model_id": self.model_id,
                    "base": self.adapter.identity.base,
                    "adapter": self.adapter.identity.adapter,
                    "quantization": self.quantization,
                    "weight_hash": self.weight_hash,
                    "config_fingerprint": self.config_fingerprint,
                    "provenance": self.provenance.model_dump(mode="json"),
                }
            )

    @property
    def identity(self) -> ModelIdentity:
        assert self.provenance is not None
        return ModelIdentity(
            base=self.adapter.identity.base,
            adapter=self.adapter.identity.adapter,
            revision=self.provenance.checkpoint_revision,
        )

    @property
    def quality_accepted(self) -> bool:
        return bool(
            self.quality_status is QualityStatus.PILOT_PASSED
            and self.quality_evidence is not None
            and self.quality_evidence.signature_verified
        )

    def public_status(self) -> dict[str, object]:
        assert self.provenance is not None
        return {
            "model_id": self.model_id,
            "base": self.adapter.identity.base,
            "adapter": self.adapter.identity.adapter,
            "state": self.state.value,
            "revision": self.provenance.checkpoint_revision,
            "runtime_ready": self.adapter.ready,
            "ready": self.adapter.ready,
            "source": self.source,
            "quantization": self.quantization,
            "weight_hash": self.weight_hash,
            "config_fingerprint": self.config_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "provenance": self.provenance.model_dump(mode="json"),
            "quality_status": self.quality_status.value,
            "quality_accepted": self.quality_accepted,
            "quality_evidence": self.quality_evidence.model_dump(mode="json"),
            "serving_tier": self.serving_tier.value if self.serving_tier else None,
        }


@dataclass(frozen=True, slots=True)
class ModelAuditRecord:
    sequence: int
    timestamp: str
    action: str
    actor: str
    request_id: str
    model_id: str
    from_state: str | None
    to_state: str | None
    previous_active: str | None
    active: str | None
    result: str
    reason: str
    config_fingerprint: str | None
    model_fingerprint: str | None
    quality_status: str
    quality_accepted: bool
    serving_tier: str | None

    def public_status(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "previous_active": self.previous_active,
            "active": self.active,
            "result": self.result,
            "reason": self.reason,
            "config_fingerprint": self.config_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "quality_status": self.quality_status,
            "quality_accepted": self.quality_accepted,
            "serving_tier": self.serving_tier,
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, ModelRegistration] = {}
        self._aliases: dict[str, str] = {}
        self._specialists: dict[str, SpecialistRegistration] = {}
        self._specialist_aliases: dict[str, str] = {}
        self._audit: deque[ModelAuditRecord] = deque(maxlen=1000)
        self._audit_sequence = 0
        self._runtime: dict[str, object] = {
            "requested_mode": "unconfigured",
            "selected_mode": "unconfigured",
            "degraded": False,
            "fallback_reason": None,
        }

    def set_runtime_status(
        self,
        *,
        requested_mode: str,
        selected_mode: str,
        degraded: bool,
        fallback_reason: str | None = None,
    ) -> None:
        with self._lock:
            self._runtime = {
                "requested_mode": requested_mode,
                "selected_mode": selected_mode,
                "degraded": degraded,
                "fallback_reason": fallback_reason,
            }

    def runtime_status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._runtime)

    def register(self, registration: ModelRegistration) -> None:
        with self._lock:
            if registration.model_id in self._models:
                raise ValueError(f"model already registered: {registration.model_id}")
            if registration.state is ModelState.ACTIVE and "active" in self._aliases:
                raise ValueError("only one model may be registered active")
            if (
                registration.state is ModelState.ACTIVE
                and registration.serving_tier is ServingTier.PRODUCTION
                and not registration.quality_accepted
            ):
                raise ValueError(
                    "production active model requires a matching signed evaluator report"
                )
            self._models[registration.model_id] = registration
            if registration.state is ModelState.ACTIVE:
                self._aliases["active"] = registration.model_id
            elif registration.state is ModelState.CANDIDATE:
                self._aliases.setdefault("candidate", registration.model_id)
            self._record_audit(
                action="register",
                actor="system",
                request_id="-",
                registration=registration,
                from_state=None,
                to_state=registration.state,
                previous_active=None,
                result="success",
                reason="registry initialization",
            )

    def set_alias(self, alias: str, model_id: str) -> None:
        with self._lock:
            if model_id not in self._models:
                raise ValueError(f"unknown model: {model_id}")
            if alias == "active" and self._models[model_id].state is not ModelState.ACTIVE:
                raise ValueError("active alias requires a model in active state")
            self._aliases[alias] = model_id

    def register_specialist(self, registration: SpecialistRegistration) -> None:
        with self._lock:
            if registration.specialist_id in self._specialists:
                raise ValueError(
                    f"specialist already registered: {registration.specialist_id}"
                )
            if registration.active and "active" in self._specialist_aliases:
                raise ValueError("only one specialist may be registered active")
            self._specialists[registration.specialist_id] = registration
            if registration.active:
                self._specialist_aliases["active"] = registration.specialist_id
            else:
                self._specialist_aliases.setdefault(
                    "candidate", registration.specialist_id
                )

    def set_specialist_alias(self, alias: str, specialist_id: str) -> None:
        with self._lock:
            if specialist_id not in self._specialists:
                raise ValueError(f"unknown specialist: {specialist_id}")
            if alias == "active" and not self._specialists[specialist_id].active:
                raise ValueError("active alias requires an active specialist")
            self._specialist_aliases[alias] = specialist_id

    def transition(
        self,
        model_id: str,
        target: ModelState,
        *,
        actor: str = "system",
        request_id: str = "-",
        reason: str = "lifecycle transition",
    ) -> ModelAuditRecord:
        if target is ModelState.VALIDATED:
            return self.validate_candidate(
                model_id, actor=actor, request_id=request_id, reason=reason
            )
        if target is ModelState.ACTIVE:
            return self.activate(model_id, actor=actor, request_id=request_id, reason=reason)
        if target is ModelState.RETIRED:
            return self.retire(model_id, actor=actor, request_id=request_id, reason=reason)
        with self._lock:
            registration = self._get(model_id)
            raise ValueError(f"invalid model state transition: {registration.state} -> {target}")

    def validate_candidate(
        self,
        model_id: str,
        *,
        actor: str,
        request_id: str,
        reason: str,
    ) -> ModelAuditRecord:
        with self._lock:
            registration = self._get(model_id)
            self._require_transition(registration, ModelState.VALIDATED)
            if not registration.adapter.ready:
                raise ValueError("candidate adapter must be ready before validation")
            if not registration.config_fingerprint or not registration.model_fingerprint:
                raise ValueError("candidate fingerprints are required before validation")
            previous_state = registration.state
            registration.state = ModelState.VALIDATED
            if self._aliases.get("candidate") == model_id:
                self._aliases.pop("candidate", None)
            return self._record_audit(
                action="validate",
                actor=actor,
                request_id=request_id,
                registration=registration,
                from_state=previous_state,
                to_state=ModelState.VALIDATED,
                previous_active=self._aliases.get("active"),
                result="success",
                reason=reason,
            )

    def activate(
        self,
        model_id: str,
        *,
        actor: str,
        request_id: str,
        reason: str,
        expected_active_model_id: str | None = None,
        expected_active_fingerprint: str | None = None,
        serving_tier: ServingTier = ServingTier.PILOT,
    ) -> ModelAuditRecord:
        """Atomically promote one validated model and preserve the old active alias."""

        with self._lock:
            registration = self._get(model_id)
            self._require_transition(registration, ModelState.ACTIVE)
            if not registration.adapter.ready:
                raise ValueError("validated adapter must be ready before activation")
            if serving_tier is ServingTier.PRODUCTION and not registration.quality_accepted:
                raise ValueError(
                    "production activation requires a matching signed evaluator report"
                )
            self._check_expected_active(expected_active_model_id, expected_active_fingerprint)
            previous_id = self._aliases.get("active")
            state_snapshot = {key: item.state for key, item in self._models.items()}
            tier_snapshot = {key: item.serving_tier for key, item in self._models.items()}
            alias_snapshot = dict(self._aliases)
            try:
                if previous_id and previous_id != model_id:
                    self._models[previous_id].state = ModelState.VALIDATED
                    self._models[previous_id].serving_tier = None
                    self._aliases["previous"] = previous_id
                registration.state = ModelState.ACTIVE
                registration.serving_tier = serving_tier
                self._aliases["active"] = model_id
                if self._aliases.get("candidate") == model_id:
                    self._aliases.pop("candidate", None)
                return self._record_audit(
                    action="activate",
                    actor=actor,
                    request_id=request_id,
                    registration=registration,
                    from_state=ModelState.VALIDATED,
                    to_state=ModelState.ACTIVE,
                    previous_active=previous_id,
                    result="success",
                    reason=reason,
                )
            except Exception:
                for key, state in state_snapshot.items():
                    self._models[key].state = state
                    self._models[key].serving_tier = tier_snapshot[key]
                self._aliases = alias_snapshot
                raise

    def rollback(
        self,
        *,
        actor: str,
        request_id: str,
        reason: str,
        expected_active_model_id: str | None = None,
        expected_active_fingerprint: str | None = None,
        serving_tier: ServingTier = ServingTier.PILOT,
    ) -> ModelAuditRecord:
        """Atomically reactivate previous and retire the rolled-back active model."""

        with self._lock:
            self._check_expected_active(expected_active_model_id, expected_active_fingerprint)
            current_id = self._aliases.get("active")
            previous_id = self._aliases.get("previous")
            if not current_id or not previous_id or current_id == previous_id:
                raise ValueError("rollback requires distinct active and previous aliases")
            current = self._get(current_id)
            previous = self._get(previous_id)
            if previous.state is not ModelState.VALIDATED or not previous.adapter.ready:
                raise ValueError("previous model must be validated and ready")
            if not previous.config_fingerprint or not previous.model_fingerprint:
                raise ValueError("previous model fingerprints are required")
            if serving_tier is ServingTier.PRODUCTION and not previous.quality_accepted:
                raise ValueError("production rollback requires a matching signed evaluator report")
            state_snapshot = {key: item.state for key, item in self._models.items()}
            tier_snapshot = {key: item.serving_tier for key, item in self._models.items()}
            alias_snapshot = dict(self._aliases)
            try:
                current.state = ModelState.RETIRED
                current.serving_tier = None
                previous.state = ModelState.ACTIVE
                previous.serving_tier = serving_tier
                self._aliases["active"] = previous_id
                self._aliases["previous"] = current_id
                if self._aliases.get("candidate") == previous_id:
                    self._aliases.pop("candidate", None)
                return self._record_audit(
                    action="rollback",
                    actor=actor,
                    request_id=request_id,
                    registration=previous,
                    from_state=ModelState.VALIDATED,
                    to_state=ModelState.ACTIVE,
                    previous_active=current_id,
                    result="success",
                    reason=reason,
                )
            except Exception:
                for key, state in state_snapshot.items():
                    self._models[key].state = state
                    self._models[key].serving_tier = tier_snapshot[key]
                self._aliases = alias_snapshot
                raise

    def retire(
        self,
        model_id: str,
        *,
        actor: str,
        request_id: str,
        reason: str,
    ) -> ModelAuditRecord:
        with self._lock:
            registration = self._get(model_id)
            if registration.state is ModelState.ACTIVE:
                raise ValueError("active model must be replaced or rolled back before retire")
            self._require_transition(registration, ModelState.RETIRED)
            previous_state = registration.state
            registration.state = ModelState.RETIRED
            registration.serving_tier = None
            for alias, aliased_model in list(self._aliases.items()):
                if aliased_model == model_id:
                    self._aliases.pop(alias, None)
            return self._record_audit(
                action="retire",
                actor=actor,
                request_id=request_id,
                registration=registration,
                from_state=previous_state,
                to_state=ModelState.RETIRED,
                previous_active=self._aliases.get("active"),
                result="success",
                reason=reason,
            )

    def _get(self, model_id: str) -> ModelRegistration:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ValueError(f"unknown model: {model_id}") from exc

    @staticmethod
    def _require_transition(registration: ModelRegistration, target: ModelState) -> None:
        if target not in ALLOWED_TRANSITIONS[registration.state]:
            raise ValueError(f"invalid model state transition: {registration.state} -> {target}")

    def _check_expected_active(
        self,
        expected_model_id: str | None,
        expected_fingerprint: str | None,
    ) -> None:
        active = self.active()
        if expected_model_id is not None and (
            active is None or active.model_id != expected_model_id
        ):
            raise ValueError("active model changed since the operation was prepared")
        if expected_fingerprint is not None and (
            active is None or active.model_fingerprint != expected_fingerprint
        ):
            raise ValueError("active model fingerprint changed")

    def _record_audit(
        self,
        *,
        action: str,
        actor: str,
        request_id: str,
        registration: ModelRegistration,
        from_state: ModelState | None,
        to_state: ModelState | None,
        previous_active: str | None,
        result: str,
        reason: str,
    ) -> ModelAuditRecord:
        self._audit_sequence += 1
        record = ModelAuditRecord(
            sequence=self._audit_sequence,
            timestamp=datetime.now(UTC).isoformat(),
            action=action,
            actor=actor,
            request_id=request_id,
            model_id=registration.model_id,
            from_state=from_state.value if from_state else None,
            to_state=to_state.value if to_state else None,
            previous_active=previous_active,
            active=self._aliases.get("active"),
            result=result,
            reason=reason,
            config_fingerprint=registration.config_fingerprint,
            model_fingerprint=registration.model_fingerprint,
            quality_status=registration.quality_status.value,
            quality_accepted=registration.quality_accepted,
            serving_tier=(registration.serving_tier.value if registration.serving_tier else None),
        )
        self._audit.append(record)
        return record

    def resolve(self, alias_or_id: str) -> ModelRegistration:
        with self._lock:
            model_id = self._aliases.get(alias_or_id, alias_or_id)
            registration = self._models.get(model_id)
            if registration is None:
                raise ServiceError(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"model alias is not registered: {alias_or_id}",
                )
            if registration.state is not ModelState.ACTIVE or not registration.adapter.ready:
                raise ServiceError(
                    ErrorCode.MODEL_NOT_READY,
                    f"model is not active and ready: {alias_or_id}",
                )
            return registration

    def active(self) -> ModelRegistration | None:
        with self._lock:
            model_id = self._aliases.get("active")
            return self._models.get(model_id) if model_id else None

    def ready(self) -> bool:
        active = self.active()
        return bool(active and active.state is ModelState.ACTIVE and active.adapter.ready)

    def specialist_active(self) -> SpecialistRegistration | None:
        with self._lock:
            specialist_id = self._specialist_aliases.get("active")
            return self._specialists.get(specialist_id) if specialist_id else None

    def resolve_specialist(self, alias_or_id: str = "active") -> SpecialistRegistration:
        with self._lock:
            specialist_id = self._specialist_aliases.get(alias_or_id, alias_or_id)
            registration = self._specialists.get(specialist_id)
            if registration is None:
                raise ServiceError(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"specialist alias is not registered: {alias_or_id}",
                )
            if not registration.active or not registration.adapter.ready:
                raise ServiceError(
                    ErrorCode.MODEL_NOT_READY,
                    f"specialist is not active and ready: {alias_or_id}",
                )
            return registration

    def mode_runtime_ready(self, mode: AnalysisMode) -> bool:
        model_ready = self.ready()
        specialist = self.specialist_active()
        specialist_ready = bool(
            specialist and specialist.active and specialist.adapter.ready
        )
        if mode is AnalysisMode.VLM_ONLY:
            return model_ready
        if mode is AnalysisMode.SPECIALIST_ONLY:
            return specialist_ready
        return model_ready and specialist_ready

    def mode_quality_status(self, mode: AnalysisMode) -> QualityStatus:
        active = self.active()
        specialist = self.specialist_active()
        statuses: list[QualityStatus] = []
        if mode in {AnalysisMode.VLM_ONLY, AnalysisMode.FUSED} and active is not None:
            statuses.append(active.quality_status)
        if mode in {AnalysisMode.SPECIALIST_ONLY, AnalysisMode.FUSED} and specialist is not None:
            statuses.append(specialist.quality_status)
        if not statuses:
            return QualityStatus.UNVALIDATED
        priority = {
            QualityStatus.PILOT_FAILED: 0,
            QualityStatus.UNVALIDATED: 1,
            QualityStatus.PILOT_CANDIDATE: 2,
            QualityStatus.PILOT_PASSED: 3,
        }
        return min(statuses, key=priority.__getitem__)

    def mode_quality_accepted(self, mode: AnalysisMode) -> bool:
        active = self.active()
        specialist = self.specialist_active()
        if mode is AnalysisMode.VLM_ONLY:
            return bool(active and active.quality_accepted)
        if mode is AnalysisMode.SPECIALIST_ONLY:
            return bool(specialist and specialist.quality_accepted)
        return bool(
            active
            and active.quality_accepted
            and specialist
            and specialist.quality_accepted
        )

    def mode_serving_tier(self, mode: AnalysisMode) -> ServingTier:
        active = self.active()
        specialist = self.specialist_active()
        tiers: list[ServingTier | None] = []
        if mode in {AnalysisMode.VLM_ONLY, AnalysisMode.FUSED}:
            tiers.append(active.serving_tier if active else None)
        if mode in {AnalysisMode.SPECIALIST_ONLY, AnalysisMode.FUSED}:
            tiers.append(specialist.serving_tier if specialist else None)
        return (
            ServingTier.PRODUCTION
            if tiers and all(item is ServingTier.PRODUCTION for item in tiers)
            else ServingTier.PILOT
        )

    def production_ready(self, mode: AnalysisMode = AnalysisMode.VLM_ONLY) -> bool:
        return bool(
            self.mode_runtime_ready(mode)
            and self.mode_serving_tier(mode) is ServingTier.PRODUCTION
            and self.mode_quality_accepted(mode)
        )

    def mode_statuses(self) -> dict[str, dict[str, object]]:
        return {
            mode.value: {
                "runtime_ready": self.mode_runtime_ready(mode),
                "quality_status": self.mode_quality_status(mode).value,
                "quality_accepted": self.mode_quality_accepted(mode),
                "serving_tier": self.mode_serving_tier(mode).value,
                "production_ready": self.production_ready(mode),
            }
            for mode in AnalysisMode
        }

    def statuses(self) -> dict[str, object]:
        with self._lock:
            return {
                "runtime": self.runtime_status(),
                "runtime_ready": self.ready(),
                "quality_accepted": bool(self.active() and self.active().quality_accepted),
                "production_ready": self.production_ready(),
                "aliases": dict(sorted(self._aliases.items())),
                "models": [self._models[key].public_status() for key in sorted(self._models)],
                "specialist_aliases": dict(sorted(self._specialist_aliases.items())),
                "specialists": [
                    self._specialists[key].public_status()
                    for key in sorted(self._specialists)
                ],
                "analysis_modes": self.mode_statuses(),
                "audit_sequence": self._audit_sequence,
            }

    def audit_records(self, *, limit: int = 100) -> list[dict[str, object]]:
        with self._lock:
            bounded = max(1, min(limit, 1000))
            return [record.public_status() for record in list(self._audit)[-bounded:]]


class MockModelAdapter:
    """Deterministic integration adapter. It contains no production algorithm."""

    def __init__(
        self,
        *,
        ready: bool = True,
        delay_seconds: float = 0.0,
        failure: str | None = None,
        base: str = "mock-vlm-0",
        adapter_id: str = "mock-compliance-v0",
    ) -> None:
        self._ready = ready
        self.delay_seconds = delay_seconds
        self.failure = failure
        self.base = base
        self.adapter_id = adapter_id

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            base=self.base,
            adapter=self.adapter_id,
            revision="mock-builtin-v0",
        )

    @property
    def ready(self) -> bool:
        return self._ready

    async def analyze(self, request: AdapterRequest) -> ModelOutput | Mapping:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.failure == "memory":
            raise MemoryError("simulated resource exhaustion")
        if self.failure == "invalid_output":
            return {"result": "violation", "objects": [{"bad": True}]}
        if self.failure == "internal":
            raise RuntimeError("simulated adapter error")

        query = request.query.casefold()
        negative = any(
            marker in query
            for marker in (
                "no violation",
                "is compliant",
                "未发现",
                "确认合规",
                "不存在目标",
            )
        )
        if negative:
            return ModelOutput(
                result="compliant",
                objects=[],
                reason="Mock adapter found no evidence; manual review is still required.",
                uncertain=False,
                warnings=["mock_adapter"],
            )

        width, height = request.image.width, request.image.height
        return ModelOutput(
            result="violation",
            objects=[
                EvidenceObject(
                    label="mock_target",
                    bbox=(width * 0.25, height * 0.25, width * 0.75, height * 0.75),
                    confidence=0.5,
                    source=ObjectSource.MOCK,
                )
            ],
            reason="Mock adapter response for API integration only.",
            uncertain=True,
            warnings=["mock_adapter", "not_for_business_decisions"],
        )


def build_mock_registry(adapter: ModelAdapter | None = None) -> ModelRegistry:
    registry = ModelRegistry()
    selected_adapter = adapter or MockModelAdapter()
    config_fingerprint = fingerprint_mapping(
        {"model": "mock-compliance-v0", "contract": "mock-contract-v0"}
    )
    registry.register(
        ModelRegistration(
            model_id="mock-compliance-v0",
            adapter=selected_adapter,
            state=ModelState.ACTIVE,
            source="built-in-test-double",
            quantization="none",
            weight_hash=None,
            config_fingerprint=config_fingerprint,
            provenance=ModelProvenance(
                model_id=selected_adapter.identity.base,
                checkpoint_revision="mock-builtin-v0",
                backend="mock",
                config_fingerprint=config_fingerprint,
                config_artifact_sha256=config_fingerprint,
                adapter_hash=None,
                weight_hash=None,
                data_version="synthetic-contract-v0",
                prompt_version="mock-contract-v0",
            ),
            quality_status=QualityStatus.UNVALIDATED,
            serving_tier=ServingTier.PILOT,
        )
    )
    registry.set_runtime_status(
        requested_mode="mock",
        selected_mode="mock",
        degraded=False,
    )
    return registry
