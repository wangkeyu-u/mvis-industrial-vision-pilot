from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import (
    GroundTruthRecord,
    normalize_analyze_response,
    normalize_model_result,
)
from src.evaluation.cli import build_records, load_ground_truth, load_predictions, main
from src.inference.contracts import (
    Decision,
    DetectedObject,
    EvidenceSource,
    ModelProvenance,
    ModelResult,
    RefusalCode,
    RefusalSignal,
)

FIXTURES = Path(__file__).parent / "fixtures"
GROUND_TRUTH = FIXTURES / "system_ground_truth.jsonl"
PREDICTIONS = FIXTURES / "system_predictions.jsonl"


class OutputAdapterTests(unittest.TestCase):
    def test_live_model_result_is_consumed_directly(self) -> None:
        result = ModelResult(
            decision=Decision.VIOLATION,
            objects=(
                DetectedObject("target", (0, 0, 100, 80), 0.9, EvidenceSource.VLM),
                DetectedObject("second", (20, 20, 30, 30), 0.8, EvidenceSource.FUSION),
            ),
            reason="two findings",
            uncertain=False,
            refusal=None,
            provenance=ModelProvenance("model-a", "r1", "mock", "fingerprint"),
        )
        prediction = normalize_model_result(result)
        self.assertTrue(prediction.schema_valid)
        self.assertEqual(prediction.result, "violation")
        self.assertEqual(len(prediction.objects), 2)
        self.assertEqual(prediction.boundary_errors(100, 80), ())

    def test_live_uncertain_model_result_preserves_refusal(self) -> None:
        result = ModelResult(
            decision=Decision.UNCERTAIN,
            objects=(),
            reason="cannot see target",
            uncertain=True,
            refusal=RefusalSignal(RefusalCode.INSUFFICIENT_EVIDENCE, "review"),
            provenance=ModelProvenance("model-a", "r1", "mock", "fingerprint"),
        )
        prediction = normalize_model_result(result)
        self.assertEqual(prediction.refusal_code, "insufficient_evidence")
        self.assertTrue(prediction.uncertain)

    def test_api_float_bbox_and_refusal_warning_are_normalized(self) -> None:
        payload = {
            "schema_version": "1.0.0",
            "request_id": "req-1",
            "model": {"base": "api-model", "adapter": None},
            "result": "uncertain",
            "objects": [],
            "reason": "not enough evidence",
            "uncertain": True,
            "latency_ms": 3,
            "warnings": ["refusal:low_confidence"],
        }
        prediction = normalize_analyze_response(payload)
        self.assertTrue(prediction.schema_valid)
        self.assertEqual(prediction.request_id, "req-1")
        self.assertEqual(prediction.refusal_code, "low_confidence")

    def test_invalid_json_is_retained_as_invalid_prediction(self) -> None:
        prediction = normalize_analyze_response("not-json")
        self.assertFalse(prediction.schema_valid)
        self.assertIn("invalid JSON", prediction.parse_error or "")

    def test_malformed_api_envelope_is_not_counted_as_valid_json(self) -> None:
        payload = {
            "schema_version": "1.0.0",
            "request_id": "",
            "model": {"base": "", "adapter": None},
            "result": "compliant",
            "objects": [],
            "reason": "no target",
            "uncertain": False,
            "latency_ms": True,
            "warnings": [1],
        }
        self.assertFalse(normalize_analyze_response(payload).schema_valid)


class BatchCliTests(unittest.TestCase):
    def test_fixture_covers_batch_metrics_slices_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            exit_code = main(
                [
                    "--ground-truth",
                    str(GROUND_TRUTH),
                    "--predictions",
                    str(PREDICTIONS),
                    "--output",
                    str(report_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        summary = report["summary"]
        self.assertEqual(summary["sample_count"], 5)
        self.assertEqual(summary["localization_target_count"], 3)
        self.assertEqual(summary["hard_negative_count"], 2)
        self.assertEqual(summary["failure_case_count"], 2)
        self.assertAlmostEqual(summary["macro_f1"], 8 / 9)
        self.assertAlmostEqual(summary["acc_at_iou"], 2 / 3)
        self.assertEqual(summary["json_schema_validity_rate"], 0.8)
        self.assertEqual(summary["hard_negative_false_positive_rate"], 0.5)
        self.assertEqual(summary["evidence_conclusion_consistency_rate"], 0.8)
        self.assertEqual(report["slice_metrics"]["difficulty:hard_negative"]["sample_count"], 2)
        self.assertEqual(report["slice_metrics"]["prediction_source:api"]["sample_count"], 4)
        self.assertEqual(
            report["slice_metrics"]["prediction_source:model_result"]["sample_count"], 1
        )

        failures = {item["sample_id"]: item["failure_codes"] for item in report["failure_cases"]}
        self.assertIn("INVALID_JSON", failures["s-invalid-json"])
        self.assertIn("HARD_NEGATIVE_FALSE_POSITIVE", failures["s-invalid-json"])
        self.assertIn("BBOX_OUT_OF_BOUNDS", failures["s-api-oob"])
        self.assertIn("LOCALIZATION_MISS", failures["s-api-oob"])
        multi = next(item for item in report["samples"] if item["sample_id"] == "s-model-multi")
        self.assertEqual(multi["metrics"]["localization_hits"], 2)

    def test_missing_prediction_becomes_a_failure_instead_of_filtering_truth(self) -> None:
        truth = GroundTruthRecord("s1", "compliant", (), 10, 10)
        records = build_records([truth], {})
        self.assertFalse(records[0].prediction.schema_valid)
        self.assertEqual(records[0].prediction.source_kind, "missing")

    def test_fixture_files_have_exactly_matching_ids(self) -> None:
        truths = load_ground_truth(GROUND_TRUTH)
        predictions = load_predictions(PREDICTIONS)
        self.assertEqual({item.sample_id for item in truths}, set(predictions))


if __name__ == "__main__":
    unittest.main()
