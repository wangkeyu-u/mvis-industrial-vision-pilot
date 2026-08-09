from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.evaluation import (
    SpecialistTileCase,
    binary_auroc,
    box_detection_metrics,
    evaluate_specialist_tiles,
    heatmap_to_bboxes,
    map_tile_bbox_to_original,
    pixel_metrics,
    threshold_heatmap,
)
from src.evaluation.specialist_cli import evaluate_prediction_file


class SpecialistMetricTests(unittest.TestCase):
    def test_heatmap_threshold_component_box_and_scaled_remap(self) -> None:
        heatmap = (
            (0.1, 0.7, 0.8, 0.0),
            (0.0, 0.9, 0.7, 0.0),
            (0.0, 0.0, 0.0, 0.8),
            (0.0, 0.0, 0.0, 0.8),
        )
        mask = threshold_heatmap(heatmap, 0.5)
        self.assertEqual(heatmap_to_bboxes(heatmap, 0.5), ((1.0, 0.0, 3.0, 2.0), (3.0, 2.0, 4.0, 4.0)))
        self.assertEqual(mask[0], (0, 1, 1, 0))
        mapped = map_tile_bbox_to_original(
            (1, 0, 3, 2),
            {
                "offset_x": 10,
                "offset_y": 20,
                "tile_width": 8,
                "tile_height": 8,
                "original_width": 30,
                "original_height": 40,
            },
            coordinate_width=4,
            coordinate_height=4,
        )
        self.assertEqual(mapped, (12.0, 20.0, 16.0, 24.0))

    def test_binary_pixel_and_box_metrics_have_explicit_empty_class_behavior(self) -> None:
        self.assertEqual(binary_auroc([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9]), 1.0)
        self.assertIsNone(binary_auroc([1, 1], [0.2, 0.8]))
        pixels = pixel_metrics([((0, 1), (0, 1))], [((0, 1), (1, 1))])
        self.assertAlmostEqual(pixels["precision"], 2 / 3)
        self.assertEqual(pixels["recall"], 1.0)
        self.assertEqual(pixels["iou"], 2 / 3)
        boxes = box_detection_metrics(
            [[(0, 0, 4, 4), (10, 10, 14, 14)]],
            [[(0, 0, 4, 4), (20, 20, 24, 24)]],
        )
        self.assertEqual(boxes["matched_boxes"], 1)
        self.assertEqual(boxes["precision"], 0.5)
        self.assertEqual(boxes["recall"], 0.5)
        self.assertEqual(boxes["acc_at_iou"], 0.5)

    def test_end_to_end_specialist_report_aggregates_tiles_to_source_images(self) -> None:
        transform = {
            "offset_x": 4,
            "offset_y": 8,
            "tile_width": 4,
            "tile_height": 4,
            "original_width": 20,
            "original_height": 20,
        }
        positive = (
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (0, 1, 1, 0),
            (0, 0, 0, 0),
        )
        negative = tuple(tuple(0 for _ in range(4)) for _ in range(4))
        cases = [
            SpecialistTileCase("pos", "source-positive", positive, positive, 0.9, transform),
            SpecialistTileCase("pos-bg", "source-positive", negative, negative, 0.2, transform),
            SpecialistTileCase("neg", "source-negative", negative, negative, 0.1, transform),
        ]
        report = evaluate_specialist_tiles(
            cases,
            heatmap_threshold=0.5,
            image_threshold=0.5,
            min_component_area=1,
            box_iou_threshold=0.5,
        )
        self.assertTrue(report["pilot_only"])
        self.assertEqual(report["model_metrics"]["image"]["f1"], 1.0)
        self.assertEqual(report["model_metrics"]["image"]["auroc"], 1.0)
        self.assertEqual(report["model_metrics"]["pixel"]["f1"], 1.0)
        self.assertEqual(report["model_metrics"]["box"]["f1"], 1.0)
        self.assertEqual(
            report["sample_results"][0]["predicted_boxes_original"],
            [[5.0, 9.0, 7.0, 11.0]],
        )
        self.assertIn("not formal KPI", report["warning"])

    def test_offline_cli_adapter_accepts_anomalib_aliases_and_low_resolution_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = Image.new("L", (4, 4), 0)
            for y in range(2):
                for x in range(2):
                    mask.putpixel((x, y), 255)
            (root / "masks").mkdir()
            mask.save(root / "masks" / "positive.png")
            transform = {
                "offset_x": 0,
                "offset_y": 0,
                "tile_width": 4,
                "tile_height": 4,
                "original_width": 4,
                "original_height": 4,
            }
            records = [
                {
                    "tile_id": "positive",
                    "source_sample_id": "positive-source",
                    "split": "test",
                    "tile_size": 4,
                    "mask": "masks/positive.png",
                    "transform": transform,
                },
                {
                    "tile_id": "negative",
                    "source_sample_id": "negative-source",
                    "split": "test",
                    "tile_size": 4,
                    "mask": None,
                    "transform": transform,
                },
            ]
            predictions = [
                {
                    "tile_id": "positive",
                    "pred_score": 0.9,
                    "heatmap": [[0.9, 0.0], [0.0, 0.0]],
                },
                {
                    "tile_id": "negative",
                    "anomaly_score": 0.1,
                    "pred_mask": [[0, 0], [0, 0]],
                },
            ]
            record_path = root / "records.jsonl"
            prediction_path = root / "predictions.jsonl"
            record_path.write_text(
                "".join(json.dumps(value) + "\n" for value in records), encoding="utf-8"
            )
            prediction_path.write_text(
                "".join(json.dumps(value) + "\n" for value in predictions), encoding="utf-8"
            )
            report = evaluate_prediction_file(
                record_path,
                root,
                prediction_path,
                split="test",
                heatmap_threshold=0.5,
                image_threshold=0.5,
            )
            self.assertEqual(report["prediction_count"], 2)
            self.assertEqual(report["model_metrics"]["image"]["auroc"], 1.0)
            self.assertEqual(report["model_metrics"]["pixel"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
