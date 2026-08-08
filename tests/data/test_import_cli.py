from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.import_cli import import_domain_samples, main
from src.data.license_policy import LicensePolicy

POLICY = Path(__file__).resolve().parents[2] / "configs" / "data" / "license_policy.yaml"


def write_pattern(path: Path, pattern: int) -> None:
    image = Image.new("L", (32, 24))
    for y in range(24):
        for x in range(32):
            image.putpixel((x, y), (x * (pattern + 5) + y * (pattern * 11 + 3) + x * y) % 256)
    image.save(path)


def source_record(index: int, license_id: str = "synthetic") -> dict:
    hard_negative = index == 7
    return {
        "entity_id": "shared-entity" if index in (2, 3) else f"entity-{index}",
        "image": f"images/{index}.png",
        "instruction": "inspect visual compliance",
        "result": "compliant" if hard_negative or index % 2 == 0 else "violation",
        "objects": [] if hard_negative or index % 2 == 0 else [{"label": "target", "bbox": [2, 2, 12, 12]}],
        "reason": "synthetic annotation",
        "hard_negative": hard_negative,
        "difficulty": ["low_light"] if index == 6 else [],
        "slices": ["location:indoor"],
        "source": "synthetic_generator",
        "license": {
            "identifier": license_id,
            "name": "Synthetic fixture license",
            "redistributable": True,
            "commercial_use": True,
            "derivative_work": True,
        },
    }


class DomainImportCliTests(unittest.TestCase):
    def test_full_synthetic_import_generates_manifest_statistics_and_data_card(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            records = []
            for index in range(8):
                write_pattern(images / f"{index}.png", index)
                records.append(source_record(index))
            input_path = root / "source.jsonl"
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            output = root / "canonical.jsonl"
            manifest = root / "manifest.json"
            summary = root / "summary.json"
            card = root / "DATA_CARD.md"
            rejects = root / "rejects.jsonl"
            exit_code = main(
                [
                    "--input", str(input_path),
                    "--dataset-root", str(root),
                    "--output", str(output),
                    "--manifest-output", str(manifest),
                    "--summary-output", str(summary),
                    "--data-card-output", str(card),
                    "--rejects-output", str(rejects),
                    "--license-policy", str(POLICY),
                    "--domain", "compliance_synth",
                    "--dataset-version", "compliance_synth-1.0.0",
                    "--created-at", "2026-08-08T00:00:00Z",
                    "--max-hash-distance", "0",
                    "--fixture-only",
                ]
            )
            self.assertEqual(exit_code, 0)
            canonical = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            stats = json.loads(summary.read_text(encoding="utf-8"))
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            card_text = card.read_text(encoding="utf-8")
            self.assertEqual(len(canonical), 8)
            self.assertTrue(all(item["sample_id"].startswith("compliance_synth_") for item in canonical))
            self.assertEqual(stats["accepted_rows"], 8)
            self.assertEqual(stats["rejected_rows"], 0)
            self.assertEqual(stats["dataset"]["hard_negative_count"], 1)
            self.assertEqual(stats["dataset"]["slice_counts"]["location:indoor"], 8)
            self.assertEqual(manifest_value["statistics"]["total"], 8)
            self.assertIn("SYNTHETIC FIXTURE", card_text)
            self.assertIn("Hard negatives: 1", card_text)
            self.assertEqual(rejects.read_text(encoding="utf-8"), "")

    def test_denied_license_is_rejected_before_dataset_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            write_pattern(root / "images" / "0.png", 0)
            input_path = root / "source.jsonl"
            input_path.write_text(json.dumps(source_record(0, "unknown")) + "\n", encoding="utf-8")
            result = import_domain_samples(
                input_path, root, LicensePolicy.load(POLICY), domain="compliance"
            )
            self.assertEqual(result.samples, ())
            self.assertEqual(result.rejected[0].code, "LICENSE_DENIED")

    def test_denylist_takes_precedence_and_policy_overlap_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "both allowed and denied"):
            LicensePolicy(frozenset({"same"}), frozenset({"same"}))


if __name__ == "__main__":
    unittest.main()
