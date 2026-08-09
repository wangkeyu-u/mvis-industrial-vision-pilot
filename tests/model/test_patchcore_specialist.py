import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.cli import load_predictions
from src.inference.patchcore_specialist import (
    build_specialist_prediction,
    heatmap_to_bbox,
    load_patchcore_config,
    select_classification_threshold,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/ksdd_patchcore_resnet18_phase7.json"


class PatchcoreSpecialistTests(unittest.TestCase):
    def test_phase7_config_pins_one_small_backbone(self) -> None:
        config = load_patchcore_config(CONFIG)

        self.assertEqual(config.backbone, "resnet18")
        self.assertLess(config.backbone_bytes, 3_000_000_000)
        self.assertEqual(config.expected_test, 56)

    def test_validation_threshold_separates_simple_scores(self) -> None:
        threshold, metrics = select_classification_threshold(
            [0.1, 0.2, 0.8, 0.9], [False, False, True, True]
        )

        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_heatmap_bbox_is_scaled_and_bounded(self) -> None:
        heatmap = np.zeros((10, 20), dtype=np.float32)
        heatmap[2:5, 4:10] = 1.0

        bbox = heatmap_to_bbox(heatmap, 0.9, 200, 100)

        self.assertEqual(bbox, (40, 20, 100, 50))

    def test_specialist_payload_is_evaluator_compatible(self) -> None:
        payload = build_specialist_prediction(
            score=0.9,
            image_threshold=0.5,
            bbox=(1, 2, 10, 12),
            confidence=0.8,
            model_id="patchcore",
            revision="a" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            import json

            path.write_text(
                json.dumps(
                    {"sample_id": "one", "source": "model_result", "prediction": payload}
                )
                + "\n",
                encoding="utf-8",
            )
            prediction = load_predictions(path)["one"]

        self.assertTrue(prediction.schema_valid)
        self.assertEqual(prediction.result, "violation")


if __name__ == "__main__":
    unittest.main()
