from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import (
    DataSample,
    DuplicateGroup,
    SchemaError,
    detect_split_leakage,
    load_samples_jsonl,
    prepare_dataset,
)


def make_sample(sample_id: str, entity_id: str, image: str, *, hard_negative: bool = False) -> DataSample:
    return DataSample.from_dict(
        {
            "sample_id": sample_id,
            "entity_id": entity_id,
            "image": image,
            "instruction": "inspect compliance",
            "response": {"result": "compliant", "objects": [], "reason": "no violation"},
            "difficulty": ["hard_negative"] if hard_negative else [],
            "source": "owned_capture",
            "license": {
                "identifier": "owned",
                "name": "Owned fixture data",
                "redistributable": False,
            },
        }
    )


def write_pattern(path: Path, pattern: int) -> None:
    image = Image.new("L", (32, 32))
    for y in range(32):
        for x in range(32):
            image.putpixel((x, y), (x * (pattern + 3) + y * (pattern * 7 + 5) + x * y) % 256)
    image.save(path)


class DatasetPipelineTests(unittest.TestCase):
    def test_prepare_dataset_closes_validation_dedup_split_manifest_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            samples = []
            for index in range(8):
                image_path = images / f"{index}.png"
                write_pattern(image_path, 0 if index == 1 else index)
                entity_id = "shared-entity" if index in (2, 3) else f"entity-{index}"
                samples.append(
                    make_sample(
                        f"s{index}", entity_id, f"images/{index}.png", hard_negative=index == 7
                    )
                )

            result = prepare_dataset(
                samples,
                root,
                dataset_version="compliance-1.0.0",
                created_at="2026-08-08T00:00:00Z",
                max_hash_distance=0,
                frozen_test=True,
            )

            self.assertTrue(result.validation.valid)
            self.assertTrue(result.leakage.passes())
            self.assertEqual(result.leakage.near_duplicate_leakage_rate, 0.0)
            self.assertEqual(result.assignments["s0"], result.assignments["s1"])
            self.assertEqual(result.assignments["s2"], result.assignments["s3"])
            self.assertEqual(result.manifest.statistics["total"], 8)
            self.assertEqual(result.manifest.statistics["hard_negative"], 1)
            self.assertEqual(len(result.manifest.entries), 8)
            self.assertEqual(set(result.assignments.values()), {"train", "validation", "test"})

    def test_jsonl_loader_reports_bad_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            sample = make_sample("s1", "e1", "images/1.png")
            path.write_text(json.dumps(sample.to_dict()) + "\nnot-json\n", encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "line 2"):
                load_samples_jsonl(path)


class LeakageAuditTests(unittest.TestCase):
    def test_reports_entity_and_near_duplicate_leakage(self) -> None:
        samples = [
            make_sample("s1", "entity-a", "images/1.png"),
            make_sample("s2", "entity-a", "images/2.png"),
            make_sample("s3", "entity-b", "images/3.png"),
        ]
        duplicate_groups = (DuplicateGroup(("s2", "s3"), 1),)
        report = detect_split_leakage(
            samples,
            {"s1": "train", "s2": "test", "s3": "validation"},
            duplicate_groups,
        )
        self.assertEqual(report.entity_leaks["entity-a"], ("test", "train"))
        self.assertEqual(report.near_duplicate_leakage_rate, 2 / 3)
        self.assertEqual(report.leaked_sample_ids, ("s2", "s3"))
        self.assertFalse(report.passes())

    def test_reports_assignment_coverage_and_invalid_split(self) -> None:
        samples = [make_sample("s1", "e1", "images/1.png"), make_sample("s2", "e2", "images/2.png")]
        report = detect_split_leakage(samples, {"s1": "holdout", "unknown": "train"})
        self.assertEqual(report.missing_sample_ids, ("s2",))
        self.assertEqual(report.unknown_assignment_ids, ("unknown",))
        self.assertEqual(report.invalid_split_sample_ids, ("s1",))
        self.assertFalse(report.passes())


if __name__ == "__main__":
    unittest.main()
