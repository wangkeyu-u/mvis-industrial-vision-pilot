import unittest

from src.training.tile_sft import tile_positions, transform_answer_for_tile


class TileSFTTests(unittest.TestCase):
    def test_positions_cover_top_middle_bottom_without_duplicates(self) -> None:
        self.assertEqual(tile_positions(1260, 500), (0, 380, 760))

    def test_bbox_is_clipped_shifted_and_scaled(self) -> None:
        answer = {
            "objects": [{"label": "surface_defect", "bbox": [50, 450, 300, 550]}]
        }

        transformed = transform_answer_for_tile(
            answer, y_offset=400, source_tile_size=500, output_size=256
        )

        self.assertEqual(transformed["result"], "violation")
        self.assertEqual(transformed["objects"][0]["bbox"], [25, 25, 154, 77])

    def test_non_overlapping_tile_becomes_compliant(self) -> None:
        answer = {
            "objects": [{"label": "surface_defect", "bbox": [50, 10, 300, 30]}]
        }

        transformed = transform_answer_for_tile(
            answer, y_offset=400, source_tile_size=500, output_size=384
        )

        self.assertEqual(transformed["result"], "compliant")
        self.assertEqual(transformed["objects"], [])


if __name__ == "__main__":
    unittest.main()
