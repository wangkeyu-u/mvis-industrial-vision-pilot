from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import (
    KSDDSourceRecord,
    LicensePolicy,
    SplitRatios,
    discover_ksdd,
    mask_bounding_box,
    prepare_ksdd_dataset,
)
from src.evaluation.cli import load_ground_truth


def write_fixture_item(root: Path, entity: int, part: int, *, defect: bool = False) -> None:
    directory = root / f"kos{entity:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (32, 24))
    for y in range(24):
        for x in range(32):
            image.putpixel((x, y), (x * (entity + 3) + y * (part + 7) + x * y) % 256)
    image.save(directory / f"Part{part}.jpg")
    mask = Image.new("L", image.size, 0)
    if defect:
        for y in range(7, 12):
            for x in range(5, 14):
                mask.putpixel((x, y), 255)
    mask.save(directory / f"Part{part}_label.bmp")


class KSDDAdapterTests(unittest.TestCase):
    def test_mask_bbox_and_physical_entity_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture_item(root, 1, 0, defect=True)
            write_fixture_item(root, 1, 1)
            items = discover_ksdd(root)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0].entity_id, items[1].entity_id)
            self.assertEqual(mask_bounding_box(items[0].mask_path).as_list(), [5, 7, 14, 12])
            self.assertIsNone(mask_bounding_box(items[1].mask_path))

    def test_real_dataset_pipeline_contract_closes_on_small_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            for entity in range(1, 9):
                write_fixture_item(source, entity, 0, defect=True)
                write_fixture_item(source, entity, 1)
            shutil.copy2(source / "kos07" / "Part0.jpg", source / "kos08" / "Part0.jpg")
            conflicting_mask = Image.new("L", (32, 24), 0)
            for y in range(8, 13):
                for x in range(6, 16):
                    conflicting_mask.putpixel((x, y), 255)
            conflicting_mask.save(source / "kos08" / "Part0_label.bmp")
            source_record = KSDDSourceRecord(
                source_page="https://www.vicos.si/resources/kolektorsdd/",
                archive_url="https://data.vicos.si/datasets/KSDD/KolektorSDD.zip",
                archive_sha256="0" * 64,
                archive_bytes=100,
                downloaded_at="2026-08-08T00:00:00Z",
            )
            result = prepare_ksdd_dataset(
                source,
                output,
                source_record,
                LicensePolicy(frozenset({"cc-by-nc-sa-4.0"}), frozenset()),
                dataset_version="ksdd_fixture-0.1.0",
                created_at="2026-08-08T00:01:00Z",
                ratios=SplitRatios(0.625, 0.125, 0.25),
                max_hash_distance=0,
                minimum_formal_kpi_test_samples=300,
            )

            self.assertEqual(len(result.samples), 15)
            self.assertEqual(result.exact_duplicate_removed_ids, ("ksdd_kos08_part0",))
            self.assertEqual(result.statistics["exact_duplicate_annotation_conflicts_merged"], 1)
            self.assertTrue(result.validation.valid)
            self.assertTrue(result.leakage.passes(0.0))
            self.assertEqual(result.leakage.near_duplicate_leakage_rate, 0.0)
            self.assertEqual(len(result.probe_sample_ids), 5)
            self.assertEqual(result.statistics["evaluation_status"], "pilot")
            self.assertFalse(result.statistics["formal_kpi_eligible"])
            self.assertEqual(result.statistics["test_hard_negative_rate"], 0.5)
            self.assertTrue(result.manifest.frozen_test)
            for entity_id in {sample.entity_id for sample in result.samples}:
                entity_splits = {
                    result.assignments[sample.sample_id]
                    for sample in result.samples
                    if sample.entity_id == entity_id
                }
                self.assertEqual(len(entity_splits), 1)
            test_negatives = [
                sample
                for sample in result.samples
                if result.assignments[sample.sample_id] == "test"
                and sample.response.result == "compliant"
            ]
            self.assertTrue(test_negatives)
            self.assertTrue(all(sample.is_hard_negative for sample in test_negatives))
            self.assertIn("PILOT DATASET", (output / "DATA_CARD.md").read_text(encoding="utf-8"))
            self.assertIn(
                "Frozen-test hard negatives: 2 (50.0%)",
                (output / "DATA_CARD.md").read_text(encoding="utf-8"),
            )
            summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
            self.assertIsNone(summary["model_metrics"])
            self.assertFalse(summary["fixture_or_mock"])
            self.assertEqual(len(json.loads((output / "probe_manifest.json").read_text())["probes"]), 5)
            truths = load_ground_truth(output / "evaluation_ground_truth.jsonl")
            self.assertEqual(len(truths), result.statistics["split_counts"]["test"])
            self.assertTrue(all(truth.image_width == 32 and truth.image_height == 24 for truth in truths))

    def test_noncommercial_license_requires_explicit_allowlist_entry(self) -> None:
        policy = LicensePolicy(frozenset({"cc-by-4.0"}), frozenset())
        with self.assertRaisesRegex(Exception, "LICENSE_NOT_ALLOWLISTED"):
            prepare_ksdd_dataset(
                ".",
                Path(tempfile.gettempdir()) / "not-created-by-license-rejection",
                KSDDSourceRecord(
                    source_page="https://example.test/source",
                    archive_url="https://example.test/archive.zip",
                    archive_sha256="0" * 64,
                    archive_bytes=1,
                    downloaded_at="2026-08-08T00:00:00Z",
                ),
                policy,
                created_at="2026-08-08T00:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
