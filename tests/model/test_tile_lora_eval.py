import unittest

from src.inference.tile_lora_eval import aggregate_tile_predictions


def record(tile_id: str, source_id: str, y_offset: int) -> dict:
    return {
        "sample_id": tile_id,
        "tile": {
            "source_sample_id": source_id,
            "source_bbox_xyxy": [0, y_offset, 500, y_offset + 500],
            "output_size": 384,
            "source_width": 500,
            "source_height": 1260,
        },
    }


def row(tile_id: str, result: str = "compliant", objects: list | None = None) -> dict:
    return {
        "sample_id": tile_id,
        "status": "completed",
        "prediction": {
            "result": result,
            "objects": objects or [],
            "reason": "reason",
            "uncertain": False,
            "refusal": None,
            "provenance": {"model_id": "test"},
            "warnings": [],
        },
    }


class TileLoraAggregationTests(unittest.TestCase):
    def test_maps_violation_bbox_to_original_coordinates(self) -> None:
        records = [record("a", "source", 0), record("b", "source", 380), record("c", "source", 760)]
        rows = [
            row("a"),
            row("b", "violation", [{"label": "surface_defect", "bbox": [38, 76, 192, 153], "confidence": 0.8, "source": "vlm"}]),
            row("c"),
        ]

        output = aggregate_tile_predictions(rows, records)[0]

        self.assertEqual(output["prediction"]["result"], "violation")
        self.assertEqual(output["prediction"]["objects"][0]["bbox"], [49, 478, 250, 580])

    def test_one_failed_tile_makes_source_unavailable(self) -> None:
        records = [record("a", "source", 0), record("b", "source", 380), record("c", "source", 760)]
        rows = [row("a"), row("b"), {"sample_id": "c", "status": "unavailable"}]

        output = aggregate_tile_predictions(rows, records)[0]

        self.assertEqual(output["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
