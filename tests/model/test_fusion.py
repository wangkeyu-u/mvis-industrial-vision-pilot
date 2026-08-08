import unittest

from src.inference import (
    ConservativeFusionPolicy,
    Decision,
    DetectedObject,
    EvidenceSource,
    ModelProvenance,
    ModelResult,
    RefusalCode,
    RefusalSignal,
)
from src.inference.contracts import SpecialistResult


def provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        model_revision="fixed-revision",
        backend="mock",
        config_fingerprint="abc123",
        extra={"request_id": "sample-1"},
    )


def violation(confidence: float = 0.90) -> ModelResult:
    return ModelResult(
        decision=Decision.VIOLATION,
        objects=(
            DetectedObject("unsafe_item", (10, 10, 40, 40), confidence),
        ),
        reason="visible finding",
        uncertain=False,
        refusal=None,
        provenance=provenance(),
    )


def compliant() -> ModelResult:
    return ModelResult(
        decision=Decision.COMPLIANT,
        objects=(),
        reason="no finding",
        uncertain=False,
        refusal=None,
        provenance=provenance(),
    )


def uncertain() -> ModelResult:
    refusal = RefusalSignal(
        RefusalCode.INSUFFICIENT_EVIDENCE, "image is unreadable"
    )
    return ModelResult(
        decision=Decision.UNCERTAIN,
        objects=(),
        reason=refusal.message,
        uncertain=True,
        refusal=refusal,
        provenance=provenance(),
    )


def specialist(
    *objects: DetectedObject,
    source: EvidenceSource = EvidenceSource.FLORENCE2,
) -> SpecialistResult:
    return SpecialistResult(source=source, objects=tuple(objects))


class FusionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ConservativeFusionPolicy()

    def test_matching_evidence_produces_traced_fused_violation(self) -> None:
        result = self.policy.fuse(
            violation(),
            specialist(
                DetectedObject(
                    "region", (12, 12, 39, 39), 0.85, EvidenceSource.FLORENCE2
                )
            ),
        )

        self.assertIs(result.decision, Decision.VIOLATION)
        self.assertEqual(result.objects[0].bbox, (12, 12, 39, 39))
        self.assertEqual(result.objects[0].confidence, 0.85)
        self.assertIs(result.objects[0].source, EvidenceSource.FUSION)
        self.assertEqual(result.provenance.extra["request_id"], "sample-1")
        self.assertEqual(result.provenance.extra["specialist_source"], "florence2")
        self.assertEqual(result.provenance.extra["matched_objects"], "1")

    def test_non_overlapping_evidence_becomes_model_conflict(self) -> None:
        result = self.policy.fuse(
            violation(),
            specialist(
                DetectedObject(
                    "region", (60, 60, 90, 90), 0.95, EvidenceSource.FLORENCE2
                )
            ),
        )

        self.assertIs(result.decision, Decision.UNCERTAIN)
        self.assertEqual(result.objects, ())
        self.assertIsNotNone(result.refusal)
        assert result.refusal is not None
        self.assertIs(result.refusal.code, RefusalCode.MODEL_CONFLICT)

    def test_compliant_primary_conflicting_with_specialist_is_reviewed(self) -> None:
        result = self.policy.fuse(
            compliant(),
            specialist(
                DetectedObject(
                    "region", (10, 10, 40, 40), 0.90, EvidenceSource.FLORENCE2
                )
            ),
        )

        self.assertIs(result.decision, Decision.UNCERTAIN)
        assert result.refusal is not None
        self.assertIs(result.refusal.code, RefusalCode.MODEL_CONFLICT)

    def test_low_confidence_primary_without_corroboration_is_refused(self) -> None:
        result = self.policy.fuse(violation(0.40), specialist())

        self.assertIs(result.decision, Decision.UNCERTAIN)
        assert result.refusal is not None
        self.assertIs(result.refusal.code, RefusalCode.LOW_CONFIDENCE)

    def test_existing_refusal_is_preserved_without_specialist_evidence(self) -> None:
        result = self.policy.fuse(uncertain(), specialist())

        self.assertIs(result.decision, Decision.UNCERTAIN)
        assert result.refusal is not None
        self.assertIs(result.refusal.code, RefusalCode.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.provenance.extra["specialist_candidates_accepted"], "0")

    def test_low_confidence_specialist_is_ignored_for_compliant_primary(self) -> None:
        result = self.policy.fuse(
            compliant(),
            specialist(
                DetectedObject(
                    "region", (10, 10, 40, 40), 0.69, EvidenceSource.FLORENCE2
                )
            ),
        )

        self.assertIs(result.decision, Decision.COMPLIANT)
        self.assertEqual(result.provenance.extra["specialist_candidates"], "1")
        self.assertEqual(result.provenance.extra["specialist_candidates_accepted"], "0")

    def test_thresholds_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "iou_threshold"):
            ConservativeFusionPolicy(iou_threshold=1.1)

    def test_multiple_specialists_must_all_corroborate(self) -> None:
        result = self.policy.fuse_many(
            violation(),
            (
                specialist(
                    DetectedObject(
                        "region", (11, 11, 40, 40), 0.88, EvidenceSource.FLORENCE2
                    )
                ),
                specialist(
                    DetectedObject(
                        "region", (12, 12, 39, 39), 0.82, EvidenceSource.RF_DETR
                    ),
                    source=EvidenceSource.RF_DETR,
                ),
            ),
        )

        self.assertIs(result.decision, Decision.VIOLATION)
        self.assertEqual(result.objects[0].confidence, 0.82)
        self.assertEqual(
            result.provenance.extra["specialist_sources"], "florence2,rf_detr"
        )

    def test_one_conflicting_specialist_makes_multi_fusion_uncertain(self) -> None:
        result = self.policy.fuse_many(
            violation(),
            (
                specialist(
                    DetectedObject(
                        "region", (11, 11, 40, 40), 0.88, EvidenceSource.FLORENCE2
                    )
                ),
                specialist(
                    DetectedObject(
                        "region", (60, 60, 90, 90), 0.92, EvidenceSource.RF_DETR
                    ),
                    source=EvidenceSource.RF_DETR,
                ),
            ),
        )

        self.assertIs(result.decision, Decision.UNCERTAIN)
        assert result.refusal is not None
        self.assertIs(result.refusal.code, RefusalCode.MODEL_CONFLICT)

    def test_multi_fusion_requires_at_least_one_specialist(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.policy.fuse_many(compliant(), ())


if __name__ == "__main__":
    unittest.main()
