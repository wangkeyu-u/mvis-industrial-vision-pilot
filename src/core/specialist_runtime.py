"""Provider-neutral anomaly specialist contracts, fusion, and heatmap storage."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image
from pydantic import BaseModel, ConfigDict

from src.core.errors import ErrorCode, ServiceError
from src.core.schemas import (
    AnalyzeOptions,
    AnalyzeTask,
    EvidenceObject,
    HeatmapArtifact,
    ImageMetadata,
    ModelIdentity,
    ModelOutput,
    ModelProvenance,
    ObjectSource,
    QualityEvidence,
    QualityStatus,
    ServingTier,
    SpecialistOutput,
)


class SpecialistRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_bytes: bytes
    image: ImageMetadata
    request_id: str
    query: str
    task: AnalyzeTask
    options: AnalyzeOptions


@runtime_checkable
class SpecialistAdapter(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    @property
    def ready(self) -> bool: ...

    async def detect(self, request: SpecialistRequest) -> SpecialistOutput | Mapping: ...


@dataclass(slots=True)
class SpecialistRegistration:
    specialist_id: str
    adapter: SpecialistAdapter
    source: str
    provenance: ModelProvenance
    quality_status: QualityStatus = QualityStatus.UNVALIDATED
    quality_evidence: QualityEvidence | None = None
    serving_tier: ServingTier = ServingTier.PILOT
    active: bool = True

    def __post_init__(self) -> None:
        if self.quality_evidence is None:
            self.quality_evidence = QualityEvidence(failure_reason="attestation_missing")
        if (
            self.active
            and self.serving_tier is ServingTier.PRODUCTION
            and not self.quality_accepted
        ):
            raise ValueError(
                "production specialist requires a matching signed evaluator report"
            )

    @property
    def quality_accepted(self) -> bool:
        return bool(
            self.quality_status is QualityStatus.PILOT_PASSED
            and self.quality_evidence is not None
            and self.quality_evidence.signature_verified
        )

    def public_status(self) -> dict[str, object]:
        assert self.quality_evidence is not None
        return {
            "specialist_id": self.specialist_id,
            "base": self.adapter.identity.base,
            "revision": self.adapter.identity.revision,
            "active": self.active,
            "runtime_ready": self.adapter.ready,
            "source": self.source,
            "provenance": self.provenance.model_dump(mode="json"),
            "quality_status": self.quality_status.value,
            "quality_accepted": self.quality_accepted,
            "quality_evidence": self.quality_evidence.model_dump(mode="json"),
            "serving_tier": self.serving_tier.value,
        }


class UnavailableSpecialistAdapter:
    def __init__(self, identity: ModelIdentity, reason_code: str) -> None:
        self._identity = identity
        self.reason_code = reason_code

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def ready(self) -> bool:
        return False

    async def detect(self, request: SpecialistRequest) -> SpecialistOutput | Mapping:
        del request
        raise ServiceError(
            ErrorCode.MODEL_NOT_READY,
            f"specialist is not ready ({self.reason_code})",
        )


class MockSpecialistAdapter:
    """Deterministic specialist test double; contains no anomaly algorithm."""

    def __init__(
        self,
        *,
        ready: bool = True,
        score: float = 0.9,
        threshold: float = 0.7,
        bbox: tuple[float, float, float, float] = (4.0, 3.0, 20.0, 15.0),
        heatmap_png: bytes | None = None,
        delay_seconds: float = 0.0,
        failure: str | None = None,
    ) -> None:
        self._ready = ready
        self.score = score
        self.threshold = threshold
        self.bbox = bbox
        self.heatmap_png = heatmap_png
        self.delay_seconds = delay_seconds
        self.failure = failure

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(base="mock-patchcore", revision="mock-specialist-v0")

    @property
    def ready(self) -> bool:
        return self._ready

    async def detect(self, request: SpecialistRequest) -> SpecialistOutput | Mapping:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.failure == "memory":
            raise MemoryError("simulated specialist resource exhaustion")
        if self.failure == "invalid_output":
            return {"score": "invalid"}
        if self.failure == "internal":
            raise RuntimeError("simulated specialist failure")
        objects = []
        if self.score >= self.threshold:
            objects.append(
                EvidenceObject(
                    label="surface_anomaly",
                    bbox=self.bbox,
                    confidence=min(1.0, self.score),
                    source=ObjectSource.PATCHCORE,
                )
            )
        return SpecialistOutput(
            score=self.score,
            threshold=self.threshold,
            source=ObjectSource.PATCHCORE,
            objects=objects,
            heatmap_png=self.heatmap_png,
            warnings=["mock_specialist", "not_for_business_decisions"],
        )


def build_mock_specialist_registration(
    adapter: SpecialistAdapter | None = None,
) -> SpecialistRegistration:
    selected = adapter or MockSpecialistAdapter()
    config_hash = hashlib.sha256(b"mock-specialist-contract-v0").hexdigest()
    return SpecialistRegistration(
        specialist_id="mock-patchcore-v0",
        adapter=selected,
        source="built-in-specialist-test-double",
        provenance=ModelProvenance(
            model_id=selected.identity.base,
            checkpoint_revision=selected.identity.revision,
            backend="mock-specialist",
            config_fingerprint=config_hash,
            config_artifact_sha256=config_hash,
            adapter_hash=None,
            weight_hash=None,
            data_version="synthetic-anomaly-v0",
            prompt_version="not-applicable",
        ),
        quality_status=QualityStatus.UNVALIDATED,
        serving_tier=ServingTier.PILOT,
    )


def specialist_only_output(specialist: SpecialistOutput) -> ModelOutput:
    if specialist.detected:
        return ModelOutput(
            result="violation",
            objects=specialist.objects,
            reason=(
                f"Specialist anomaly score {specialist.score:.6g} met threshold "
                f"{specialist.threshold:.6g}; localization is specialist-owned."
            ),
            uncertain=False,
            warnings=[*specialist.warnings, "specialist_localization_authoritative"],
        )
    return ModelOutput(
        result="compliant",
        objects=[],
        reason=(
            f"Specialist anomaly score {specialist.score:.6g} was below threshold "
            f"{specialist.threshold:.6g}."
        ),
        uncertain=False,
        warnings=specialist.warnings,
    )


def conservative_fuse(
    primary: ModelOutput,
    specialist: SpecialistOutput,
    *,
    iou_threshold: float = 0.5,
) -> tuple[ModelOutput, bool]:
    """Fuse decisions while making specialist localization exclusively authoritative."""

    primary_positive = primary.result.strip().lower() == "violation" and bool(primary.objects)
    primary_negative = primary.result.strip().lower() in {
        "compliant",
        "no_violation",
        "not_found",
        "none",
        "negative",
    }
    warnings = list(dict.fromkeys([*primary.warnings, *specialist.warnings]))
    if specialist.detected:
        agrees = primary_positive and _geometrically_agrees(
            primary.objects,
            specialist.objects,
            iou_threshold=iou_threshold,
        )
        if agrees:
            return (
                ModelOutput(
                    result="violation",
                    objects=specialist.objects,
                    reason=primary.reason,
                    uncertain=False,
                    warnings=warnings
                    + [
                        "fused_evidence_agreement",
                        "specialist_localization_authoritative",
                    ],
                ),
                False,
            )
        return (
            ModelOutput(
                result="uncertain",
                objects=specialist.objects,
                reason=(
                    "VLM decision or localization conflicts with positive specialist evidence; "
                    "specialist boxes are retained and human review is required."
                ),
                uncertain=True,
                warnings=warnings
                + [
                    "model_conflict",
                    "specialist_localization_authoritative",
                    "human_review_required",
                ],
            ),
            True,
        )

    if primary_negative:
        return (
            ModelOutput(
                result="compliant",
                objects=[],
                reason=primary.reason,
                uncertain=False,
                warnings=warnings + ["fused_negative_agreement"],
            ),
            False,
        )
    return (
        ModelOutput(
            result="uncertain",
            objects=[],
            reason=(
                "VLM positive or uncertain output lacks specialist localization; "
                "VLM boxes are discarded and human review is required."
            ),
            uncertain=True,
            warnings=warnings
            + [
                "model_conflict",
                "vlm_localization_discarded",
                "human_review_required",
            ],
        ),
        True,
    )


def _geometrically_agrees(
    primary: list[EvidenceObject],
    specialist: list[EvidenceObject],
    *,
    iou_threshold: float,
) -> bool:
    if len(primary) != len(specialist):
        return False
    remaining = set(range(len(specialist)))
    for item in primary:
        ranked = sorted(
            (
                (_iou(item.bbox, specialist[index].bbox), index)
                for index in remaining
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < iou_threshold:
            return False
        remaining.remove(ranked[0][1])
    return not remaining


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


@dataclass(frozen=True, slots=True)
class _StoredHeatmap:
    content: bytes
    artifact: HeatmapArtifact
    expires_at: float


class HeatmapArtifactStore:
    """Small in-memory, expiring store; adapter paths never cross the API boundary."""

    def __init__(
        self,
        *,
        max_items: int = 32,
        max_bytes: int = 4 * 1024 * 1024,
        ttl_seconds: int = 300,
        max_pixels: int = 4_194_304,
    ) -> None:
        if min(max_items, max_bytes, ttl_seconds, max_pixels) <= 0:
            raise ValueError("heatmap store limits must be positive")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.max_pixels = max_pixels
        self._lock = threading.RLock()
        self._items: OrderedDict[str, _StoredHeatmap] = OrderedDict()

    def add(self, content: bytes | None) -> HeatmapArtifact | None:
        if content is None:
            return None
        width, height = self._validate_png(content)
        artifact_id = f"hm_{uuid.uuid4().hex}"
        artifact = HeatmapArtifact(
            artifact_id=artifact_id,
            uri=f"/v1/artifacts/heatmaps/{artifact_id}",
            sha256=hashlib.sha256(content).hexdigest(),
            width=width,
            height=height,
            expires_in_seconds=self.ttl_seconds,
        )
        with self._lock:
            self._prune()
            self._items[artifact_id] = _StoredHeatmap(
                content=content,
                artifact=artifact,
                expires_at=time.monotonic() + self.ttl_seconds,
            )
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return artifact

    def get(self, artifact_id: str) -> bytes | None:
        with self._lock:
            self._prune()
            stored = self._items.get(artifact_id)
            if stored is None:
                return None
            self._items.move_to_end(artifact_id)
            return stored.content

    def _prune(self) -> None:
        now = time.monotonic()
        for key, item in list(self._items.items()):
            if item.expires_at <= now:
                self._items.pop(key, None)

    def _validate_png(self, content: bytes) -> tuple[int, int]:
        if not content or len(content) > self.max_bytes:
            raise ValueError("heatmap PNG exceeds the bounded artifact size")
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("heatmap artifact must be a single-frame PNG")
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError("heatmap artifact is not a valid PNG") from exc
        if width <= 0 or height <= 0 or width * height > self.max_pixels:
            raise ValueError("heatmap artifact dimensions exceed the bounded pixel budget")
        return width, height
