import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from src.evaluation.cli import load_predictions
from src.inference import MockBackend, Qwen3VLAdapter, load_model_config
from src.inference.ksdd_baseline import (
    BaselineSample,
    BaselineSpec,
    load_baseline_spec,
    load_frozen_test_samples,
    run_baseline,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json"
SPEC = ROOT / "configs/models/ksdd_zero_shot_prompt_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class KSDDBaselineTests(unittest.TestCase):
    def test_repository_spec_is_pinned_and_deterministic(self) -> None:
        spec = load_baseline_spec(SPEC)

        self.assertEqual(spec.expected_test_samples, 56)
        self.assertTrue(spec.generation.deterministic)
        self.assertEqual(spec.generation.seed, 20260808)
        self.assertEqual(len(spec.query_sha256), 64)

    def test_manifest_and_sample_images_are_cross_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "one.png"
            image.parent.mkdir()
            Image.new("RGB", (16, 12), "white").save(image)
            record = {
                "sample_id": "one",
                "image": "images/one.png",
                "metadata": {"split": "test"},
            }
            samples_path = root / "samples.jsonl"
            samples_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest = {
                "dataset_version": "ksdd-0.1.0",
                "frozen_test": True,
                "entries": [
                    {
                        "sample_id": "one",
                        "image": "images/one.png",
                        "sha256": sha256(image),
                        "split": "test",
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            spec = replace(
                load_baseline_spec(SPEC),
                expected_test_samples=1,
                dataset_manifest_sha256=sha256(manifest_path),
            )

            loaded = load_frozen_test_samples(spec, manifest_path, samples_path)

        self.assertEqual(loaded[0].sample_id, "one")
        self.assertEqual((loaded[0].image_width, loaded[0].image_height), (16, 12))

    def test_mock_full_population_export_is_evaluator_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "one.png"
            Image.new("RGB", (64, 48), "white").save(image)
            original = load_baseline_spec(SPEC)
            spec = BaselineSpec(
                original.schema_version,
                original.experiment_id,
                original.prompt_id,
                original.dataset_version,
                original.dataset_manifest_sha256,
                1,
                original.model_config,
                original.model_id,
                original.model_revision,
                original.query,
                original.generation,
            )
            sample = (BaselineSample("one", image, sha256(image), 64, 48),)
            adapter = Qwen3VLAdapter(load_model_config(MODEL_CONFIG), MockBackend())

            report = run_baseline(adapter, sample, spec, root / "output")
            predictions = load_predictions(root / "output/predictions.jsonl")

        self.assertEqual(report["counts"], {"total": 1, "completed": 1, "unavailable": 0})
        self.assertTrue(predictions["one"].schema_valid)
        self.assertEqual(report["performance"]["samples"][0]["status"], "completed")

    def test_population_mismatch_stops_before_model_load(self) -> None:
        spec = load_baseline_spec(SPEC)
        adapter = Qwen3VLAdapter(load_model_config(MODEL_CONFIG), MockBackend())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "population"):
                run_baseline(adapter, (), spec, directory)


if __name__ == "__main__":
    unittest.main()
