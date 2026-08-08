from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.deduplication import (
    DuplicateGroup,
    cross_split_leakage_rate,
    find_near_duplicate_groups,
    perceptual_hash,
)
from src.data.schema import DataSample
from src.data.splitting import assert_isolated, entity_isolated_split
from src.data.validation import validate_dataset, validate_image


def make_sample(sample_id: str, entity_id: str, image: str = "images/test.png") -> DataSample:
    return DataSample.from_dict(
        {
            "sample_id": sample_id,
            "entity_id": entity_id,
            "image": image,
            "instruction": "inspect",
            "response": {"result": "compliant", "objects": [], "reason": "no violation"},
            "difficulty": [],
            "source": "owned_capture",
            "license": {"identifier": "owned", "name": "Owned data"},
        }
    )


class ImageValidationTests(unittest.TestCase):
    def test_decodes_image_and_rejects_out_of_bounds_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            path = root / "images" / "test.png"
            Image.new("RGB", (16, 12), "white").save(path)
            metadata = validate_image(path)
            self.assertEqual((metadata.width, metadata.height, metadata.format), (16, 12, "PNG"))

            payload = make_sample("s1", "e1").to_dict()
            payload["response"]["result"] = "violation"
            payload["response"]["objects"] = [{"label": "target", "bbox": [1, 1, 17, 10]}]
            report = validate_dataset([DataSample.from_dict(payload)], root)
            self.assertFalse(report.valid)
            self.assertEqual(report.errors[0].code, "BBOX_OUT_OF_BOUNDS")

    def test_rejects_corrupt_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.png"
            path.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "cannot be decoded"):
                validate_image(path)

    def test_rejects_extension_disguised_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actually-png.jpg"
            Image.new("RGB", (8, 8), "white").save(path, format="PNG")
            with self.assertRaisesRegex(ValueError, "extension"):
                validate_image(path)


class DeduplicationAndSplitTests(unittest.TestCase):
    def test_detects_duplicate_images_and_cross_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.png", root / "second.png"
            Image.new("RGB", (16, 16), "white").save(first)
            Image.new("RGB", (16, 16), "white").save(second)
            groups = find_near_duplicate_groups({"s1": perceptual_hash(first), "s2": perceptual_hash(second)})
            self.assertEqual(groups[0].sample_ids, ("s1", "s2"))
            self.assertEqual(cross_split_leakage_rate(groups, {"s1": "train", "s2": "test"}, 2), 1.0)

    def test_split_keeps_entities_and_duplicate_components_together(self) -> None:
        samples = [
            make_sample("s1", "entity-a"),
            make_sample("s2", "entity-a"),
            make_sample("s3", "entity-b"),
            make_sample("s4", "entity-c"),
            make_sample("s5", "entity-d"),
            make_sample("s6", "entity-e"),
        ]
        duplicate_groups = (DuplicateGroup(("s3", "s4"), 2),)
        first = entity_isolated_split(samples, duplicate_groups, seed=7)
        second = entity_isolated_split(samples, duplicate_groups, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["s1"], first["s2"])
        self.assertEqual(first["s3"], first["s4"])
        assert_isolated(samples, first, duplicate_groups)


if __name__ == "__main__":
    unittest.main()
