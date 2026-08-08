import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.cli import load_predictions
from src.inference import (
    BatchPredictionSample,
    MockBackend,
    ModelAdapter,
    ModelRequest,
    Qwen3VLAdapter,
    export_predictions_jsonl,
    load_model_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json"


def sample(sample_id: str, query: str) -> BatchPredictionSample:
    return BatchPredictionSample(
        sample_id,
        ModelRequest(
            image=object(), image_width=100, image_height=80, query=query
        ),
    )


class FailingAdapter(ModelAdapter):
    @property
    def ready(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def analyze(self, request: ModelRequest):
        raise RuntimeError("raw backend output must not leak")


class PredictionExportTests(unittest.TestCase):
    def test_completed_predictions_are_evaluator_compatible(self) -> None:
        adapter = Qwen3VLAdapter(load_model_config(CONFIG), MockBackend())
        samples = (
            sample("violation", "find violation"),
            sample("uncertain", "insufficient evidence"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            summary = export_predictions_jsonl(adapter, samples, path)
            predictions = load_predictions(path)
            lines = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(summary.completed, 2)
        self.assertEqual(summary.unavailable, 0)
        self.assertTrue(all(value.schema_valid for value in predictions.values()))
        self.assertEqual(predictions["violation"].result, "violation")
        self.assertEqual(predictions["uncertain"].refusal_code, "insufficient_evidence")
        self.assertTrue(all(line["source"] == "model_result" for line in lines))

    def test_failure_is_explicit_unavailable_and_not_a_fake_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            summary = export_predictions_jsonl(
                FailingAdapter(), (sample("failed", "check"),), path
            )
            text = path.read_text(encoding="utf-8")
            prediction = load_predictions(path)["failed"]

        self.assertEqual(summary.completed, 0)
        self.assertEqual(summary.unavailable, 1)
        self.assertFalse(prediction.schema_valid)
        self.assertIn('"status": "unavailable"', text)
        self.assertNotIn("raw backend output", text)

    def test_duplicate_sample_ids_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            with self.assertRaisesRegex(ValueError, "unique"):
                export_predictions_jsonl(
                    FailingAdapter(),
                    (sample("same", "one"), sample("same", "two")),
                    path,
                )
            self.assertFalse(path.exists())

    def test_fail_fast_does_not_replace_existing_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("previous\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                export_predictions_jsonl(
                    FailingAdapter(),
                    (sample("failed", "check"),),
                    path,
                    fail_fast=True,
                )
            self.assertEqual(path.read_text(encoding="utf-8"), "previous\n")


if __name__ == "__main__":
    unittest.main()
