from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import (
    bootstrap_confidence_intervals,
    compare_model_reports,
    evaluate_kpi_acceptance,
    evaluate_offline_records,
    export_evaluation_package,
)
from src.evaluation.cli import build_records, load_ground_truth, load_predictions, main

FIXTURES = Path(__file__).parent / "fixtures"
GROUND_TRUTH = FIXTURES / "system_ground_truth.jsonl"
PREDICTIONS = FIXTURES / "system_predictions.jsonl"
DATA_MANIFEST = FIXTURES / "dataset_manifest.json"
BASELINE = FIXTURES / "baseline_metrics.json"
BASELINE_PREDICTIONS = FIXTURES / "baseline_system_predictions.jsonl"


def fixture_report():
    return evaluate_offline_records(
        build_records(load_ground_truth(GROUND_TRUTH), load_predictions(PREDICTIONS))
    )


def baseline_report():
    return evaluate_offline_records(
        build_records(load_ground_truth(GROUND_TRUTH), load_predictions(BASELINE_PREDICTIONS))
    )


class ConfidenceIntervalTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_preserves_denominators(self) -> None:
        report = fixture_report()
        first = bootstrap_confidence_intervals(report, resamples=200, seed=7)
        second = bootstrap_confidence_intervals(report, resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["macro_f1"].observation_count, 5)
        self.assertEqual(first["acc_at_iou"].observation_count, 3)
        self.assertEqual(first["hard_negative_false_positive_rate"].observation_count, 2)
        for interval in first.values():
            self.assertIsNotNone(interval.lower)
            self.assertIsNotNone(interval.upper)
            assert interval.lower is not None and interval.upper is not None
            self.assertLessEqual(interval.lower, interval.upper)


class KpiAcceptanceTests(unittest.TestCase):
    def test_all_kpis_can_pass_with_required_baselines(self) -> None:
        metrics = {
            "macro_f1": 0.90,
            "acc_at_iou": 0.80,
            "json_schema_validity_rate": 1.0,
            "hard_negative_false_positive_rate": 0.05,
            "evidence_conclusion_consistency_rate": 0.98,
        }
        baseline = {
            "macro_f1": 0.84,
            "acc_at_iou": 0.71,
            "hard_negative_false_positive_rate": 0.10,
        }
        acceptance = evaluate_kpi_acceptance(metrics, baseline_metrics=baseline)
        self.assertEqual(acceptance.status, "passed")
        self.assertTrue(acceptance.eligible_for_model_acceptance)
        self.assertTrue(all(item.status == "passed" for item in acceptance.assessments))

    def test_missing_baselines_are_not_silently_accepted(self) -> None:
        metrics = {
            "macro_f1": 0.90,
            "acc_at_iou": 0.80,
            "json_schema_validity_rate": 1.0,
            "hard_negative_false_positive_rate": 0.05,
            "evidence_conclusion_consistency_rate": 0.98,
        }
        acceptance = evaluate_kpi_acceptance(metrics)
        self.assertEqual(acceptance.status, "not_evaluable")
        self.assertFalse(acceptance.eligible_for_model_acceptance)
        keyed = {item.kpi_id: item for item in acceptance.assessments}
        self.assertEqual(keyed["KPI-01"].status, "not_evaluable")
        self.assertEqual(keyed["KPI-02"].status, "not_evaluable")
        self.assertEqual(keyed["KPI-04"].status, "not_evaluable")
        self.assertEqual(keyed["KPI-03"].status, "passed")
        self.assertEqual(keyed["KPI-05"].status, "passed")

    def test_fixture_metrics_are_never_model_acceptance_results(self) -> None:
        acceptance = evaluate_kpi_acceptance(fixture_report().summary, fixture_only=True)
        self.assertEqual(acceptance.status, "fixture_only")
        self.assertFalse(acceptance.eligible_for_model_acceptance)
        self.assertTrue(all(item.status == "fixture_only" for item in acceptance.assessments))
        self.assertTrue(
            all(item.reasons[0].code == "FIXTURE_NOT_MODEL_SCORE" for item in acceptance.assessments)
        )

    def test_mock_metrics_are_never_model_acceptance_results(self) -> None:
        acceptance = evaluate_kpi_acceptance(fixture_report().summary, mock_only=True)
        self.assertEqual(acceptance.status, "mock_only")
        self.assertFalse(acceptance.eligible_for_model_acceptance)
        self.assertTrue(all(item.status == "mock_only" for item in acceptance.assessments))
        self.assertTrue(
            all(item.reasons[0].code == "MOCK_NOT_MODEL_SCORE" for item in acceptance.assessments)
        )

    def test_pilot_metrics_are_never_formal_kpi_acceptance_results(self) -> None:
        acceptance = evaluate_kpi_acceptance(fixture_report().summary, pilot_only=True)
        self.assertEqual(acceptance.status, "pilot_only")
        self.assertFalse(acceptance.eligible_for_model_acceptance)
        self.assertTrue(all(item.status == "pilot_only" for item in acceptance.assessments))
        self.assertTrue(
            all(
                item.reasons[0].code == "PILOT_DATASET_BELOW_SAMPLE_FLOOR"
                for item in acceptance.assessments
            )
        )


class EvaluationPackageTests(unittest.TestCase):
    def test_package_contains_provenance_intervals_failures_and_checksums(self) -> None:
        report = fixture_report()
        comparison = compare_model_reports(
            baseline_report(), report, resamples=200, seed=7, minimum_reliable_samples=30
        ).to_dict()
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "package-a"
            second_path = Path(directory) / "package-b"
            first = export_evaluation_package(
                report,
                first_path,
                resolved_config={"iou_threshold": 0.5, "source": "contract_fixture"},
                data_manifest_path=DATA_MANIFEST,
                created_at="2026-08-08T00:00:00Z",
                baseline_metrics=baseline,
                bootstrap_resamples=200,
                bootstrap_seed=7,
                fixture_only=True,
                comparison=comparison,
            )
            second = export_evaluation_package(
                report,
                second_path,
                resolved_config={"iou_threshold": 0.5, "source": "contract_fixture"},
                data_manifest_path=DATA_MANIFEST,
                created_at="2026-08-08T00:00:00Z",
                baseline_metrics=baseline,
                bootstrap_resamples=200,
                bootstrap_seed=7,
                fixture_only=True,
                comparison=comparison,
            )
            self.assertEqual(first.component_hashes, second.component_hashes)
            package_manifest = json.loads(
                (first_path / "package_manifest.json").read_text(encoding="utf-8")
            )
            provenance = json.loads(
                (first_path / "data_provenance.json").read_text(encoding="utf-8")
            )
            metrics = json.loads((first_path / "metrics.json").read_text(encoding="utf-8"))
            failures = json.loads((first_path / "failures.json").read_text(encoding="utf-8"))
            environment = json.loads(
                (first_path / "environment.json").read_text(encoding="utf-8")
            )
            markdown_report = (first_path / "report.md").read_text(encoding="utf-8")
            html_report = (first_path / "report.html").read_text(encoding="utf-8")

            expected_manifest_hash = hashlib.sha256(DATA_MANIFEST.read_bytes()).hexdigest()
            self.assertEqual(provenance["manifest_sha256"], expected_manifest_hash)
            self.assertTrue(package_manifest["fixture_only"])
            self.assertIn("must not be reported as model performance", package_manifest["fixture_notice"])
            self.assertEqual(metrics["kpi_acceptance"]["status"], "fixture_only")
            self.assertEqual(
                metrics["confidence_intervals"]["acc_at_iou"]["observation_count"], 3
            )
            self.assertEqual(failures["by_code"]["INVALID_JSON"]["count"], 1)
            self.assertEqual(failures["by_slice"]["prediction_source:api"]["sample_count"], 4)
            self.assertIn("python", environment)
            self.assertIn("SYNTHETIC FIXTURE", markdown_report)
            self.assertIn("SYNTHETIC FIXTURE", html_report)
            self.assertIn("KPI-01 to KPI-05 acceptance", markdown_report)
            self.assertIn("Candidate model comparison", markdown_report)
            self.assertIn("degraded_small_sample", markdown_report)
            self.assertIn("Candidate model comparison", html_report)
            self.assertTrue((first_path / "comparison.json").is_file())
            self.assertEqual(len(package_manifest["component_sha256"]), 9)
            for filename, expected_hash in package_manifest["component_sha256"].items():
                self.assertEqual(hashlib.sha256((first_path / filename).read_bytes()).hexdigest(), expected_hash)

    def test_cli_exports_fixture_only_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exit_code = main(
                [
                    "--ground-truth",
                    str(GROUND_TRUTH),
                    "--predictions",
                    str(PREDICTIONS),
                    "--output",
                    str(root / "report.json"),
                    "--package-dir",
                    str(root / "package"),
                    "--data-manifest",
                    str(DATA_MANIFEST),
                    "--baseline-metrics",
                    str(BASELINE),
                    "--created-at",
                    "2026-08-08T00:00:00Z",
                    "--bootstrap-resamples",
                    "100",
                    "--bootstrap-seed",
                    "7",
                    "--fixture-only",
                ]
            )
            self.assertEqual(exit_code, 0)
            package_manifest = json.loads(
                (root / "package" / "package_manifest.json").read_text(encoding="utf-8")
            )
            config = json.loads(
                (root / "package" / "config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(package_manifest["fixture_only"])
            self.assertEqual(config["ground_truth"]["filename"], GROUND_TRUTH.name)
            self.assertEqual(len(config["ground_truth"]["sha256"]), 64)
            self.assertEqual(config["predictions"]["filename"], PREDICTIONS.name)
            self.assertEqual(len(config["predictions"]["sha256"]), 64)

    def test_mock_package_has_distinct_watermark_and_acceptance_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_path = Path(directory) / "mock-package"
            export_evaluation_package(
                fixture_report(),
                package_path,
                resolved_config={"source": "mock"},
                data_manifest_path=DATA_MANIFEST,
                created_at="2026-08-08T00:00:00Z",
                bootstrap_resamples=100,
                bootstrap_seed=7,
                mock_only=True,
            )
            markdown = (package_path / "report.md").read_text(encoding="utf-8")
            html = (package_path / "report.html").read_text(encoding="utf-8")
            metrics = json.loads((package_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("MOCK OUTPUT", markdown)
            self.assertIn("MOCK OUTPUT", html)
            self.assertEqual(metrics["kpi_acceptance"]["status"], "mock_only")
            self.assertFalse(metrics["kpi_acceptance"]["eligible_for_model_acceptance"])

    def test_manifest_pilot_gate_watermarks_package_and_blocks_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
            manifest["statistics"] = {
                "evaluation_status": "pilot",
                "formal_kpi_eligible": False,
                "formal_kpi_ineligibility_reason": "test split has 56 samples; minimum is 300",
            }
            pilot_manifest = root / "pilot-manifest.json"
            pilot_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            package_path = root / "pilot-package"
            exported = export_evaluation_package(
                fixture_report(),
                package_path,
                resolved_config={"source": "real_pilot"},
                data_manifest_path=pilot_manifest,
                created_at="2026-08-08T00:00:00Z",
                bootstrap_resamples=100,
                bootstrap_seed=7,
            )
            metrics = json.loads((package_path / "metrics.json").read_text(encoding="utf-8"))
            provenance = json.loads(
                (package_path / "data_provenance.json").read_text(encoding="utf-8")
            )
            self.assertTrue(exported.pilot_only)
            self.assertEqual(metrics["kpi_acceptance"]["status"], "pilot_only")
            self.assertFalse(metrics["kpi_acceptance"]["eligible_for_model_acceptance"])
            self.assertFalse(provenance["formal_kpi_eligible"])
            self.assertIn("PILOT DATASET", (package_path / "report.md").read_text())
            self.assertIn("PILOT DATASET", (package_path / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()
