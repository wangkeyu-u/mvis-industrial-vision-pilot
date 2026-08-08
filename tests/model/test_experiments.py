import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.training.experiments import (
    EvidenceStatus,
    EvidenceValue,
    ExperimentKind,
    RunManifest,
    RunStatus,
    generate_default_matrix,
    main,
    matrix_payload,
    validate_experiment_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json"


class ExperimentMatrixTests(unittest.TestCase):
    def test_default_matrix_covers_all_preregistered_axes(self) -> None:
        specs = generate_default_matrix(CONFIG)

        self.assertEqual(len(specs), 24)
        self.assertEqual(len({spec.experiment_id for spec in specs}), 24)
        self.assertEqual(len({spec.fingerprint for spec in specs}), 24)
        self.assertEqual({spec.kind for spec in specs}, set(ExperimentKind))
        self.assertTrue(all(spec.status.value == "not_run" for spec in specs))

    def test_dry_run_separates_schema_validity_from_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unavailable = validate_experiment_matrix(
                generate_default_matrix(CONFIG), cache_root=directory
            )
            config = json.loads(CONFIG.read_text(encoding="utf-8"))
            snapshot = (
                Path(directory)
                / "models--mlx-community--Qwen3-VL-2B-Instruct-4bit"
                / "snapshots"
                / config["revision"]
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"test-placeholder")
            runnable = validate_experiment_matrix(
                generate_default_matrix(CONFIG, data_version="domain-v1"),
                cache_root=directory,
                available_specialists=("florence2", "rf_detr"),
                available_quantization_bits=(4, 8, 16),
                available_adaptations=("lora", "qlora"),
            )

        self.assertTrue(unavailable.valid)
        self.assertFalse(unavailable.runnable)
        self.assertIn("frozen data version is unavailable", unavailable.blockers)
        self.assertTrue(
            any("pinned model snapshot unavailable" in item for item in unavailable.blockers)
        )
        self.assertTrue(
            any("specialist implementations unavailable" in item for item in unavailable.blockers)
        )
        self.assertTrue(
            any("quantization artifacts unavailable" in item for item in unavailable.blockers)
        )
        self.assertTrue(
            any("adaptation implementations unavailable" in item for item in unavailable.blockers)
        )
        self.assertTrue(runnable.valid)
        self.assertTrue(runnable.runnable)

    def test_payload_contains_only_explicit_not_run_evidence(self) -> None:
        payload = matrix_payload(generate_default_matrix(CONFIG))

        self.assertEqual(payload["status"], "not_run")
        for experiment in payload["experiments"]:
            manifest = experiment["manifest"]
            self.assertEqual(manifest["status"], "not_run")
            for section in ("metrics", "resources", "artifacts"):
                for evidence in manifest[section].values():
                    self.assertEqual(evidence["status"], "not_run")
                    self.assertIsNone(evidence["value"])

    def test_manifest_records_known_data_version_without_claiming_execution(self) -> None:
        spec = generate_default_matrix(CONFIG, data_version="domain-v1")[0]
        manifest = RunManifest.not_run(spec).to_dict()

        self.assertEqual(manifest["data_version"]["status"], "recorded")
        self.assertEqual(manifest["data_version"]["value"], "domain-v1")
        self.assertEqual(manifest["execution"]["started_at"]["status"], "not_run")

    def test_spec_fingerprint_does_not_depend_on_machine_local_config_path(self) -> None:
        spec = generate_default_matrix(CONFIG)[0]
        relocated = replace(spec, base_config_path="/another/machine/model.json")

        self.assertEqual(spec.fingerprint, relocated.fingerprint)

    def test_duplicate_and_incomplete_matrix_is_invalid(self) -> None:
        spec = generate_default_matrix(CONFIG)[0]
        report = validate_experiment_matrix((spec, replace(spec)))

        self.assertFalse(report.valid)
        self.assertTrue(any("duplicate" in error for error in report.errors))
        self.assertTrue(any("missing required" in error for error in report.errors))

    def test_evidence_value_prevents_fabricated_measurements(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            EvidenceValue(EvidenceStatus.NOT_RUN, value=0.95)
        with self.assertRaisesRegex(ValueError, "requires a value"):
            EvidenceValue(EvidenceStatus.RECORDED)
        with self.assertRaisesRegex(ValueError, "requires a reason"):
            EvidenceValue(EvidenceStatus.UNAVAILABLE)

    def test_completed_manifest_requires_recorded_execution_times(self) -> None:
        spec = generate_default_matrix(CONFIG)[0]
        manifest = RunManifest.not_run(spec)

        with self.assertRaisesRegex(ValueError, "recorded start and end"):
            replace(manifest, run_id="run-1", status=RunStatus.COMPLETED)

    def test_cli_marks_missing_data_as_not_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix.json"
            # Capture stdout through the standard library without adding a CLI
            # output-file mutation contract.
            import contextlib

            with output.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(
                stream
            ):
                exit_code = main(
                    ["--config", str(CONFIG), "--require-runnable"]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["dry_run"]["runnable"])


if __name__ == "__main__":
    unittest.main()
