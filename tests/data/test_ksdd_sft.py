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
    export_ksdd_sft,
    load_sft_jsonl,
    prepare_ksdd_dataset,
)
from src.data.deduplication import sha256_file
from src.data.ksdd_sft import TEST_PROMPT_VARIANT, TRAIN_PROMPT_VARIANTS

ROOT = Path(__file__).resolve().parents[2]
RECORD_SCHEMA = ROOT / "configs" / "data" / "ksdd_sft_record.schema.json"
ANSWER_SCHEMA = ROOT / "configs" / "data" / "ksdd_sft_answer.schema.json"


def write_item(root: Path, entity: int, part: int, *, defect: bool) -> None:
    directory = root / f"kos{entity:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (40, 30))
    for y in range(30):
        for x in range(40):
            image.putpixel((x, y), (x * (entity + 5) + y * (part + 11) + x * y) % 256)
    image.save(directory / f"Part{part}.jpg")
    mask = Image.new("L", image.size, 0)
    if defect:
        for y in range(8, 14):
            for x in range(6, 17):
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


class KSDDSFTExportTests(unittest.TestCase):
    def test_export_preserves_source_split_and_strict_answer_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = prepare_source(root)
            output = root / "ksdd_sft_v1"
            source_hash = sha256_file(source / "manifest.json")
            result = export_ksdd_sft(
                source,
                output,
                expected_source_manifest_sha256=source_hash,
                created_at="2026-08-09T00:01:00Z",
                record_schema_path=RECORD_SCHEMA,
                answer_schema_path=ANSWER_SCHEMA,
                dataset_version="ksdd_sft_fixture-1.0.0",
                prompt_version="ksdd_audit_prompt-1.0.0",
                reference_root=root,
            )

            train = load_sft_jsonl(output / "hf" / "train.jsonl")
            valid = load_sft_jsonl(output / "hf" / "valid.jsonl")
            test = load_sft_jsonl(output / "hf" / "test.jsonl")
            all_records = [*train, *valid, *test]
            self.assertEqual(len(all_records), 16)
            self.assertEqual(result.split_counts, {"train": 10, "valid": 2, "test": 4})
            self.assertEqual({row["prompt_variant"] for row in test}, {TEST_PROMPT_VARIANT})
            self.assertEqual({row["prompt_variant"] for row in valid}, {TEST_PROMPT_VARIANT})
            self.assertTrue({row["prompt_variant"] for row in train} <= set(TRAIN_PROMPT_VARIANTS))
            self.assertGreater(len({row["prompt_variant"] for row in train}), 1)
            self.assertEqual(
                {row["sample_id"] for row in train} & {row["sample_id"] for row in test},
                set(),
            )
            self.assertEqual(
                {row["entity_id"] for row in train} & {row["entity_id"] for row in test},
                set(),
            )
            for record in all_records:
                answer = json.loads(record["messages"][1]["content"][0]["text"])
                self.assertFalse(answer["uncertain"])
                if answer["result"] == "compliant":
                    self.assertEqual(answer["objects"], [])
                else:
                    self.assertEqual(answer["objects"][0]["bbox"], [6, 8, 17, 14])
                self.assertEqual(record["messages"][0]["content"][0]["image"], record["images"][0])

            manifest = json.loads((output / "sft_manifest.json").read_text(encoding="utf-8"))
            leakage = json.loads((output / "leakage_report.json").read_text(encoding="utf-8"))
            readiness = json.loads(
                (output / "pilot_readiness_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["source_manifest_sha256"], source_hash)
            self.assertEqual(manifest["statistics"]["strict_answer_json_validity"], 1.0)
            self.assertEqual(
                manifest["statistics"]["prompt_variant_counts_by_split"]["test"],
                {TEST_PROMPT_VARIANT: 4},
            )
            self.assertEqual(manifest["statistics"]["test_derived_training_sample_count"], 0)
            self.assertFalse(manifest["statistics"]["formal_kpi_eligible"])
            self.assertIsNone(manifest["model_metrics"])
            self.assertEqual(leakage["train_test_sample_overlap"], [])
            self.assertEqual(leakage["train_test_entity_overlap"], [])
            self.assertEqual(leakage["train_test_image_sha256_overlap"], [])
            self.assertTrue(leakage["passes"])
            self.assertTrue(readiness["pilot_only"])
            self.assertFalse(readiness["model_run"])
            self.assertIsNone(readiness["metrics"]["zero_shot"])
            self.assertEqual(
                len(load_sft_jsonl(output / "evaluation_ground_truth.jsonl")),
                len(test),
            )
            self.assertEqual(
                sha256_file(output / "sft_manifest.json"),
                result.manifest_sha256,
            )
            for relative, digest in manifest["file_sha256"].items():
                self.assertEqual(sha256_file(output / relative), digest)

    def test_wrong_source_manifest_hash_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = prepare_source(root)
            output = root / "must_not_exist"
            with self.assertRaisesRegex(Exception, "source manifest hash mismatch"):
                export_ksdd_sft(
                    source,
                    output,
                    expected_source_manifest_sha256="f" * 64,
                    created_at="2026-08-09T00:01:00Z",
                    record_schema_path=RECORD_SCHEMA,
                    answer_schema_path=ANSWER_SCHEMA,
                )
            self.assertFalse(output.exists())

    def test_schemas_are_valid_json_and_encode_negative_and_test_constraints(self) -> None:
        record_schema = json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))
        answer_schema = json.loads(ANSWER_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(record_schema["properties"]["split"]["enum"], ["train", "valid", "test"])
        self.assertEqual(answer_schema["properties"]["uncertain"]["const"], False)
        self.assertEqual(answer_schema["allOf"][0]["then"]["properties"]["objects"]["maxItems"], 0)


if __name__ == "__main__":
    unittest.main()
