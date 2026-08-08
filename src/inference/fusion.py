"""Conservative fusion of primary VLM and specialist detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import (
    Decision,
    DetectedObject,
    EvidenceSource,
    ModelProvenance,
    ModelResult,
    RefusalCode,
    RefusalSignal,
    SpecialistResult,
)


def _intersection_over_union(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


@dataclass(frozen=True, slots=True)
class ConservativeFusionPolicy:
    """Require geometrically consistent evidence before emitting a violation.

    A specialist never converts an uncertain primary answer directly into a
    positive finding. Conflicting or incomplete evidence is routed to review.
    """

    specialist_confidence_threshold: float = 0.70
    primary_confidence_threshold: float = 0.50
    iou_threshold: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "specialist_confidence_threshold",
            "primary_confidence_threshold",
            "iou_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def fuse(self, primary: ModelResult, specialist: SpecialistResult) -> ModelResult:
        candidates = tuple(
            item
            for item in specialist.objects
            if item.confidence >= self.specialist_confidence_threshold
        )
        trace = {
            "fusion_policy": type(self).__name__,
            "specialist_source": specialist.source.value,
            "specialist_candidates": str(len(specialist.objects)),
            "specialist_candidates_accepted": str(len(candidates)),
        }

        if primary.decision is Decision.UNCERTAIN:
            if candidates:
                return self._uncertain(
                    primary,
                    RefusalCode.MODEL_CONFLICT,
                    "Primary model refused while specialist produced positive evidence.",
                    trace,
                    "specialist evidence conflicts with primary refusal",
                )
            return self._copy(
                primary,
                trace,
                "specialist supplied no high-confidence corroboration",
            )

        if primary.decision is Decision.COMPLIANT:
            if candidates:
                return self._uncertain(
                    primary,
                    RefusalCode.MODEL_CONFLICT,
                    "Primary compliant decision conflicts with specialist evidence.",
                    trace,
                    "specialist evidence conflicts with compliant decision",
                )
            return self._copy(
                primary,
                trace,
                "specialist supplied no high-confidence positive evidence",
            )

        matches = self._match(primary.objects, candidates)
        if len(matches) == len(primary.objects) == len(candidates):
            fused = tuple(
                DetectedObject(
                    label=vlm_object.label,
                    bbox=specialist_object.bbox,
                    confidence=min(vlm_object.confidence, specialist_object.confidence),
                    source=EvidenceSource.FUSION,
                )
                for vlm_object, specialist_object in matches
            )
            provenance = self._provenance(primary, trace | {"matched_objects": str(len(fused))})
            return ModelResult(
                decision=Decision.VIOLATION,
                objects=fused,
                reason="Primary and specialist evidence agree geometrically.",
                uncertain=False,
                refusal=None,
                provenance=provenance,
                warnings=primary.warnings + ("violation corroborated by specialist",),
                raw_text=primary.raw_text,
            )

        if not candidates and all(
            item.confidence < self.primary_confidence_threshold for item in primary.objects
        ):
            return self._uncertain(
                primary,
                RefusalCode.LOW_CONFIDENCE,
                "Primary evidence is below confidence threshold and lacks corroboration.",
                trace,
                "low-confidence primary evidence was not corroborated",
            )

        return self._uncertain(
            primary,
            RefusalCode.MODEL_CONFLICT,
            "Primary and specialist evidence do not agree geometrically.",
            trace | {"matched_objects": str(len(matches))},
            "primary and specialist evidence conflict",
        )

    def fuse_many(
        self,
        primary: ModelResult,
        specialists: Sequence[SpecialistResult],
    ) -> ModelResult:
        """Fuse independent specialists without silently favoring one source."""

        if not specialists:
            raise ValueError("at least one specialist result is required")
        if len(specialists) == 1:
            return self.fuse(primary, specialists[0])

        results = tuple(self.fuse(primary, item) for item in specialists)
        sources = ",".join(item.source.value for item in specialists)
        trace = {
            "fusion_policy": type(self).__name__,
            "specialist_sources": sources,
            "specialist_result_count": str(len(specialists)),
        }
        if all(result.decision is Decision.VIOLATION for result in results):
            fused_objects: list[DetectedObject] = []
            for corroborated in zip(*(result.objects for result in results), strict=True):
                conservative = min(corroborated, key=lambda item: item.confidence)
                fused_objects.append(
                    DetectedObject(
                        label=conservative.label,
                        bbox=conservative.bbox,
                        confidence=conservative.confidence,
                        source=EvidenceSource.FUSION,
                    )
                )
            return ModelResult(
                decision=Decision.VIOLATION,
                objects=tuple(fused_objects),
                reason="Primary and all specialist sources agree geometrically.",
                uncertain=False,
                refusal=None,
                provenance=self._provenance(
                    primary, trace | {"matched_objects": str(len(fused_objects))}
                ),
                warnings=primary.warnings
                + ("violation corroborated by all specialist sources",),
                raw_text=primary.raw_text,
            )

        if all(result.decision is Decision.COMPLIANT for result in results):
            return self._copy(
                primary,
                trace,
                "all specialist sources supplied no high-confidence positive evidence",
            )

        conflict = any(
            result.refusal is not None
            and result.refusal.code is RefusalCode.MODEL_CONFLICT
            for result in results
        )
        code = RefusalCode.MODEL_CONFLICT if conflict else results[0].refusal.code
        reason = (
            "Primary and specialist sources contain conflicting evidence."
            if conflict
            else results[0].reason
        )
        return self._uncertain(
            primary,
            code,
            reason,
            trace,
            "multi-specialist evidence requires human review",
        )

    def _match(
        self,
        primary: tuple[DetectedObject, ...],
        specialist: tuple[DetectedObject, ...],
    ) -> tuple[tuple[DetectedObject, DetectedObject], ...]:
        available = set(range(len(specialist)))
        matches: list[tuple[DetectedObject, DetectedObject]] = []
        for vlm_object in primary:
            ranked = sorted(
                (
                    (_intersection_over_union(vlm_object.bbox, specialist[index].bbox), index)
                    for index in available
                ),
                reverse=True,
            )
            if not ranked or ranked[0][0] < self.iou_threshold:
                continue
            _, index = ranked[0]
            available.remove(index)
            matches.append((vlm_object, specialist[index]))
        return tuple(matches)

    @staticmethod
    def _provenance(primary: ModelResult, trace: dict[str, str]) -> ModelProvenance:
        original = primary.provenance
        return ModelProvenance(
            model_id=original.model_id,
            model_revision=original.model_revision,
            backend=original.backend,
            config_fingerprint=original.config_fingerprint,
            adapter_id=original.adapter_id,
            seed=original.seed,
            deterministic=original.deterministic,
            extra=dict(original.extra) | trace,
        )

    def _copy(
        self, primary: ModelResult, trace: dict[str, str], warning: str
    ) -> ModelResult:
        return ModelResult(
            decision=primary.decision,
            objects=primary.objects,
            reason=primary.reason,
            uncertain=primary.uncertain,
            refusal=primary.refusal,
            provenance=self._provenance(primary, trace),
            warnings=primary.warnings + (warning,),
            raw_text=primary.raw_text,
        )

    def _uncertain(
        self,
        primary: ModelResult,
        code: RefusalCode,
        reason: str,
        trace: dict[str, str],
        warning: str,
    ) -> ModelResult:
        return ModelResult(
            decision=Decision.UNCERTAIN,
            objects=(),
            reason=reason,
            uncertain=True,
            refusal=RefusalSignal(code=code, message=reason),
            provenance=self._provenance(primary, trace),
            warnings=primary.warnings + (warning,),
            raw_text=primary.raw_text,
        )
