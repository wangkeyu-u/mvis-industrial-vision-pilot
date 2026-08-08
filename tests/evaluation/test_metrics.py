from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.evaluation import EvaluationCase, evaluate_cases
from src.evaluation.metrics import (
    acc_at_iou,
    evidence_conclusion_consistency_rate,
    hard_negative_false_positive_rate,
    intersection_over_union,
    json_schema_validity_rate,
    macro_f1,
    validate_prediction_payload,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "evaluation_cases.json").read_text())


class MetricFixtureTests(unittest.TestCase):
    def test_macro_f1(self) -> None:
        cases = FIXTURE["classification"]
        self.assertAlmostEqual(macro_f1(cases["targets"], cases["predictions"]), cases["expected_macro_f1"])

    def test_iou_and_acc_threshold_is_inclusive(self) -> None:
        cases = FIXTURE["localization"]
        self.assertEqual(intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(acc_at_iou(cases["predictions"], cases["targets"]), cases["expected_acc_at_iou_0_5"])
        self.assertEqual(acc_at_iou([[0, 0, 10, 5]], [[0, 0, 10, 10]], threshold=0.5), 1.0)

    def test_json_validity(self) -> None:
        self.assertEqual(json_schema_validity_rate(FIXTURE["outputs"]), FIXTURE["expected_json_validity"])
        self.assertFalse(validate_prediction_payload("not-json"))

    def test_hard_negative_false_positive_rate(self) -> None:
        self.assertEqual(
            hard_negative_false_positive_rate(FIXTURE["outputs"], FIXTURE["hard_negative_mask"]),
            FIXTURE["expected_hard_negative_fpr"],
        )

    def test_evidence_conclusion_consistency(self) -> None:
        self.assertEqual(
            evidence_conclusion_consistency_rate(FIXTURE["outputs"]), FIXTURE["expected_consistency"]
        )
        uncertain = {"result": "uncertain", "objects": [], "reason": "insufficient evidence", "uncertain": True}
        self.assertEqual(evidence_conclusion_consistency_rate([uncertain]), 1.0)

    def test_malformed_hard_negative_is_counted_conservatively(self) -> None:
        self.assertEqual(hard_negative_false_positive_rate(["not-json"], [True]), 1.0)

    def test_unified_evaluator_preserves_metric_denominators(self) -> None:
        classification = FIXTURE["classification"]
        localization = FIXTURE["localization"]
        cases = []
        for index, (target, prediction, output, hard_negative) in enumerate(
            zip(
                classification["targets"],
                classification["predictions"],
                FIXTURE["outputs"],
                FIXTURE["hard_negative_mask"],
            )
        ):
            cases.append(
                EvaluationCase(
                    sample_id=f"s{index}",
                    target_label=target,
                    predicted_label=prediction,
                    raw_output=output,
                    is_hard_negative=hard_negative,
                    target_bbox=localization["targets"][index] if index < 2 else None,
                    predicted_bbox=localization["predictions"][index] if index < 2 else None,
                )
            )
        summary = evaluate_cases(cases)
        self.assertEqual(summary.sample_count, 4)
        self.assertEqual(summary.localization_count, 2)
        self.assertEqual(summary.hard_negative_count, 2)
        self.assertAlmostEqual(summary.macro_f1, classification["expected_macro_f1"])
        self.assertEqual(summary.acc_at_iou, localization["expected_acc_at_iou_0_5"])
        self.assertEqual(summary.json_schema_validity_rate, FIXTURE["expected_json_validity"])
        self.assertEqual(
            summary.hard_negative_false_positive_rate, FIXTURE["expected_hard_negative_fpr"]
        )
        self.assertEqual(
            summary.evidence_conclusion_consistency_rate, FIXTURE["expected_consistency"]
        )

    def test_empty_slices_are_explicit_zero(self) -> None:
        self.assertEqual(macro_f1([], []), 0.0)
        self.assertEqual(acc_at_iou([], []), 0.0)
        self.assertEqual(json_schema_validity_rate([]), 0.0)
        self.assertEqual(hard_negative_false_positive_rate([], []), 0.0)
        self.assertEqual(evidence_conclusion_consistency_rate([]), 0.0)


if __name__ == "__main__":
    unittest.main()
