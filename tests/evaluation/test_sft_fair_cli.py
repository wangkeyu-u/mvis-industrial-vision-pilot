from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.sft_fair_cli import main

FIXTURES = Path(__file__).parent / "fixtures"
GROUND_TRUTH = FIXTURES / "system_ground_truth.jsonl"
ZERO_PREDICTIONS = FIXTURES / "baseline_system_predictions.jsonl"
LORA_PREDICTIONS = FIXTURES / "system_predictions.jsonl"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_config(run_id: str, test_hash: str, *, adapter: bool) -> dict:
    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "dataset_version": "ksdd_sft-1.0.0",
        "prompt_version": "ksdd_audit_prompt-1.0.0",
        "test_jsonl_sha256": test_hash,
        "base_model_revision": "9c4f5209e57b31f4b9dfba735de3fb983739c9cc",
        "model_config_sha256": "1" * 64,
        "generation": {
            "seed": 20260809,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
        },
        "adapter": {"enabled": adapter, "sha256": "2" * 64 if adapter else None},
    }


class SFTFairEvaluationCliTests(unittest.TestCase):
    def test_fixture_comparison_requires_same_identity_and_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_hash = hashlib.sha256(b"frozen-test").hexdigest()
            manifest = root / "manifest.json"
            zero_config = root / "zero.json"
            lora_config = root / "lora.json"
            write_json(
                manifest,
                {
                    "dataset_version": "ksdd_sft-1.0.0",
                    "prompt_version": "ksdd_audit_prompt-1.0.0",
                    "prompt_policy": {"test_prompt_fixed": True},
                    "file_sha256": {"hf/test.jsonl": test_hash},
                    "statistics": {
                        "evaluation_status": "pilot",
                        "formal_kpi_eligible": False,
                    },
                },
            )
            write_json(zero_config, run_config("zero", test_hash, adapter=False))
            write_json(lora_config, run_config("lora", test_hash, adapter=True))
            output = root / "fair-output"
            exit_code = main(
                [
                    "--ground-truth",
                    str(GROUND_TRUTH),
                    "--zero-shot-predictions",
                    str(ZERO_PREDICTIONS),
                    "--lora-predictions",
                    str(LORA_PREDICTIONS),
                    "--zero-shot-run-config",
                    str(zero_config),
                    "--lora-run-config",
                    str(lora_config),
                    "--data-manifest",
                    str(manifest),
                    "--output-dir",
                    str(output),
                    "--created-at",
                    "2026-08-09T00:00:00Z",
                    "--bootstrap-resamples",
                    "100",
                    "--bootstrap-seed",
                    "7",
                    "--fixture-only",
                ]
            )
            self.assertEqual(exit_code, 0)
            fair = json.loads((output / "fair_report.json").read_text(encoding="utf-8"))
            zero_metrics = json.loads(
                (output / "zero_shot_package" / "metrics.json").read_text(encoding="utf-8")
            )
            lora_metrics = json.loads(
                (output / "lora_package" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue(fair["fair_comparison"])
            self.assertTrue(fair["fixture_only"])
            self.assertEqual(fair["comparison"]["conclusion_strength"], "degraded_small_sample")
            self.assertEqual(zero_metrics["kpi_acceptance"]["status"], "fixture_only")
            self.assertEqual(lora_metrics["kpi_acceptance"]["status"], "fixture_only")

    def test_generation_mismatch_blocks_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_hash = "3" * 64
            manifest = root / "manifest.json"
            zero_config = root / "zero.json"
            lora_config = root / "lora.json"
            write_json(
                manifest,
                {
                    "dataset_version": "ksdd_sft-1.0.0",
                    "prompt_version": "ksdd_audit_prompt-1.0.0",
                    "prompt_policy": {"test_prompt_fixed": True},
                    "file_sha256": {"hf/test.jsonl": test_hash},
                    "statistics": {"formal_kpi_eligible": False},
                },
            )
            zero = run_config("zero", test_hash, adapter=False)
            lora = run_config("lora", test_hash, adapter=True)
            lora["generation"]["max_tokens"] = 512
            write_json(zero_config, zero)
            write_json(lora_config, lora)
            with self.assertRaisesRegex(ValueError, "identical generation"):
                main(
                    [
                        "--ground-truth",
                        str(GROUND_TRUTH),
                        "--zero-shot-predictions",
                        str(ZERO_PREDICTIONS),
                        "--lora-predictions",
                        str(LORA_PREDICTIONS),
                        "--zero-shot-run-config",
                        str(zero_config),
                        "--lora-run-config",
                        str(lora_config),
                        "--data-manifest",
                        str(manifest),
                        "--output-dir",
                        str(root / "output"),
                        "--created-at",
                        "2026-08-09T00:00:00Z",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
