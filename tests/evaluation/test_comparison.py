from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.evaluation import FairComparisonError, compare_model_reports, evaluate_offline_records
from src.evaluation.cli import build_records, load_ground_truth, load_predictions
from src.evaluation.compare_cli import main

FIXTURES = Path(__file__).parent / "fixtures"
GROUND_TRUTH = FIXTURES / "system_ground_truth.jsonl"
BASELINE = FIXTURES / "baseline_system_predictions.jsonl"
CANDIDATE = FIXTURES / "system_predictions.jsonl"


def reports():
    truths = load_ground_truth(GROUND_TRUTH)
    baseline = evaluate_offline_records(build_records(truths, load_predictions(BASELINE)))
    candidate = evaluate_offline_records(build_records(truths, load_predictions(CANDIDATE)))
    return baseline, candidate


class FairModelComparisonTests(unittest.TestCase):
    def test_small_fixture_is_explicitly_degraded(self) -> None:
        baseline, candidate = reports()
        comparison = compare_model_reports(
            baseline, candidate, resamples=200, seed=7, minimum_reliable_samples=30
        )
        self.assertTrue(comparison.fair_comparison)
        self.assertEqual(comparison.sample_count, 5)
        self.assertEqual(comparison.conclusion_strength, "degraded_small_sample")
        self.assertIn("do not claim statistical significance", comparison.notice)
        self.assertTrue(
            all(item.significance_hint == "insufficient_sample" for item in comparison.metrics.values())
        )
        self.assertGreater(comparison.metrics["macro_f1"].delta, 0)
        self.assertGreater(comparison.metrics["acc_at_iou"].delta, 0)
        self.assertGreater(
            comparison.metrics["hard_negative_false_positive_rate"].improvement_delta, 0
        )

    def test_mismatched_population_is_rejected(self) -> None:
        baseline, candidate = reports()
        candidate = replace(candidate, samples=candidate.samples[:-1])
        with self.assertRaisesRegex(FairComparisonError, "different sample IDs"):
            compare_model_reports(baseline, candidate, resamples=100)

    def test_comparison_cli_writes_machine_readable_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            exit_code = main(
                [
                    "--ground-truth", str(GROUND_TRUTH),
                    "--baseline-predictions", str(BASELINE),
                    "--candidate-predictions", str(CANDIDATE),
                    "--output", str(output),
                    "--bootstrap-resamples", "100",
                    "--bootstrap-seed", "7",
                    "--minimum-reliable-samples", "30",
                ]
            )
            self.assertEqual(exit_code, 0)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["conclusion_strength"], "degraded_small_sample")
            self.assertEqual(value["metrics"]["macro_f1"]["significance_hint"], "insufficient_sample")


if __name__ == "__main__":
    unittest.main()
