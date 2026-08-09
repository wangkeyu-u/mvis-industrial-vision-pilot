from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import (
    KSDDSourceRecord,
    LicensePolicy,
    SplitRatios,
    export_ksdd_tiles,
    load_tile_records,
    prepare_ksdd_dataset,
)
from src.data.deduplication import sha256_file

ROOT = Path(__file__).resolve().parents[2]
TILE_SCHEMA = ROOT / "configs" / "data" / "ksdd_tile_record.schema.json"
TILE_SFT_SCHEMA = ROOT / "configs" / "data" / "ksdd_tile_sft_record.schema.json"
SFT_ANSWER_SCHEMA = ROOT / "configs" / "data" / "ksdd_sft_answer.schema.json"


def write_item(root: Path, entity: int, part: int, *, defect: bool) -> None:
    directory = root / f"kos{entity:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (48, 40))
    for y in range(40):
        for x in range(48):
            image.putpixel((x, y), (x * (entity + 7) + y * (part + 13) + x * y) % 256)
    image.save(directory / f"Part{part}.jpg")
    mask = Image.new("L", image.size, 0)
    if defect:
        for y in range(3, 9):
            for x in range(4, 38):
                mask.putpixel((x, y), 255)
    mask.save(directory / f"Part{part}_label.bmp")


def prepare_source(root: Path) -> Path:
    raw = root / "raw"
    source = root / "ksdd_v0"
    for entity in range(1, 9):
        write_item(raw, entity, 0, defect=True)
        write_item(raw, entity, 1, defect=False)
    prepare_ksdd_dataset(
        raw,
        source,
        KSDDSourceRecord(
            source_page="https://www.vicos.si/resources/kolektorsdd/",
            archive_url="https://data.vicos.si/datasets/KSDD/KolektorSDD.zip",
            archive_sha256="0" * 64,
            archive_bytes=100,
            downloaded_at="2026-08-09T00:00:00Z",
        ),
        LicensePolicy(frozenset({"cc-by-nc-sa-4.0"}), frozenset()),
        dataset_version="ksdd_fixture-0.1.0",
        created_at="2026-08-09T00:00:00Z",
        ratios=SplitRatios(0.625, 0.125, 0.25),
        max_hash_distance=0,
    )
    return source


class KSDDTileExportTests(unittest.TestCase):
    def test_export_closes_tile_sft_anomalib_and_leakage_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = prepare_source(root)
            output = root / "tiles"
            result = export_ksdd_tiles(
                source,
                output,
                expected_source_manifest_sha256=sha256_file(source / "manifest.json"),
                created_at="2026-08-09T01:00:00Z",
                record_schema_path=TILE_SCHEMA,
                sft_record_schema_path=TILE_SFT_SCHEMA,
                sft_answer_schema_path=SFT_ANSWER_SCHEMA,
                dataset_version="ksdd_tiles_fixture-1.0.0",
                prompt_version="ksdd_tile_prompt-1.0.0",
                tile_sizes=(16, 24),
                reference_root=root,
            )
            records = load_tile_records(output / "tile_records.jsonl")
            self.assertTrue(records)
            self.assertEqual({row["tile_size"] for row in records}, {16, 24})
            self.assertTrue(result.pilot_only)
            self.assertTrue(result.leakage_report["passes"])
            self.assertEqual(result.leakage_report["test_derived_training_tile_count"], 0)

            source_splits: dict[str, str] = {}
            split_entities: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
            for row in records:
                existing = source_splits.setdefault(row["source_sample_id"], row["split"])
                self.assertEqual(existing, row["split"])
                split_entities[row["split"]].add(row["entity_id"])
                with Image.open(output / row["image"]) as tile:
                    self.assertEqual(tile.size, (row["tile_size"], row["tile_size"]))
                if row["label"] == "defect":
                    self.assertIsNone(row["negative_source"])
                    self.assertIsNotNone(row["bbox_tile"])
                    self.assertIsNotNone(row["bbox_original"])
                    transform = row["transform"]
                    self.assertEqual(
                        row["bbox_original"],
                        [
                            row["bbox_tile"][0] + transform["offset_x"],
                            row["bbox_tile"][1] + transform["offset_y"],
                            row["bbox_tile"][2] + transform["offset_x"],
                            row["bbox_tile"][3] + transform["offset_y"],
                        ],
                    )
                    with Image.open(output / row["mask"]) as mask:
                        self.assertIsNotNone(mask.getbbox())
                else:
                    self.assertIn(
                        row["negative_source"],
                        {"same_image_nondefect_region", "normal_image"},
                    )
                    self.assertIsNone(row["mask"])
                    self.assertIsNone(row["bbox_tile"])
            self.assertFalse(split_entities["train"] & split_entities["test"])

            manifest = json.loads((output / "tile_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["statistics"]["positive_source_size_coverage_min"], 1.0)
            self.assertEqual(
                manifest["statistics"]["positive_source_bbox_reconstruction_rate"], 1.0
            )
            self.assertEqual(manifest["statistics"]["evaluation_status"], "pilot_only")
            self.assertFalse(manifest["statistics"]["formal_kpi_eligible"])
            self.assertIsNone(manifest["model_metrics"])
            self.assertEqual(sha256_file(output / "tile_manifest.json"), result.tile_manifest_sha256)

            for size in (16, 24):
                csv_path = output / "anomalib" / str(size) / "samples.csv"
                self.assertTrue(csv_path.is_file())
                header = csv_path.read_text(encoding="utf-8").splitlines()[0]
                for field in ("image_path", "split", "label_index", "mask_path"):
                    self.assertIn(field, header)
                for split in ("train", "valid", "test"):
                    sft_rows = load_tile_records(
                        output / "mlx_vlm" / str(size) / "hf" / f"{split}.jsonl"
                    )
                    self.assertTrue(sft_rows)
                    for sft in sft_rows:
                        answer = json.loads(sft["messages"][1]["content"][0]["text"])
                        self.assertFalse(answer["uncertain"])
                        if answer["result"] == "compliant":
                            self.assertEqual(answer["objects"], [])
                        else:
                            bbox = answer["objects"][0]["bbox"]
                            self.assertLess(bbox[0], bbox[2])
                            self.assertLess(bbox[1], bbox[3])
                    if split == "test":
                        self.assertEqual(
                            {row["prompt_variant"] for row in sft_rows},
                            {"tile_bilingual_direct"},
                        )

    def test_wrong_frozen_source_hash_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = prepare_source(root)
            output = root / "must_not_exist"
            with self.assertRaisesRegex(Exception, "source manifest hash mismatch"):
                export_ksdd_tiles(
                    source,
                    output,
                    expected_source_manifest_sha256="f" * 64,
                    created_at="2026-08-09T01:00:00Z",
                    record_schema_path=TILE_SCHEMA,
                    sft_record_schema_path=TILE_SFT_SCHEMA,
                    sft_answer_schema_path=SFT_ANSWER_SCHEMA,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
