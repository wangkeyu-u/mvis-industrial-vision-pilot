from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.manifest import DatasetManifest, ManifestEntry, build_manifest
from src.data.schema import DataSample, LicenseInfo, SchemaError


def sample_payload() -> dict:
    return {
        "sample_id": "compliance_train_000001",
        "entity_id": "site_a_camera_1",
        "image": "images/000001.png",
        "instruction": "找出不合规区域",
        "response": {
            "result": "violation",
            "objects": [{"label": "blocked_exit", "bbox": [1, 2, 8, 9]}],
            "reason": "安全出口被遮挡",
        },
        "difficulty": ["occlusion"],
        "source": "owned_capture",
        "license": {
            "identifier": "proprietary-review-only",
            "name": "Internal review license",
            "redistributable": False,
            "commercial_use": True,
        },
    }


class SampleSchemaTests(unittest.TestCase):
    def test_round_trip_and_default_uncertain(self) -> None:
        sample = DataSample.from_dict(sample_payload())
        self.assertFalse(sample.response.uncertain)
        self.assertEqual(sample.response.objects[0].bbox.as_list(), [1, 2, 8, 9])
        self.assertEqual(DataSample.from_dict(sample.to_dict()), sample)

    def test_rejects_path_traversal_and_invalid_box(self) -> None:
        payload = sample_payload()
        payload["image"] = "../secret.png"
        with self.assertRaisesRegex(SchemaError, "relative path"):
            DataSample.from_dict(payload)
        payload = sample_payload()
        payload["response"]["objects"][0]["bbox"] = [2, 2, 2, 9]
        with self.assertRaisesRegex(SchemaError, "x1 < x2"):
            DataSample.from_dict(payload)


class ManifestTests(unittest.TestCase):
    def test_atomic_round_trip_preserves_license_registry(self) -> None:
        license_info = LicenseInfo("cc-by-4.0", "Creative Commons Attribution 4.0")
        entry = ManifestEntry(
            sample_id="s1",
            entity_id="e1",
            image="images/1.png",
            split="test",
            sha256="a" * 64,
            perceptual_hash="0" * 16,
            source="licensed_public_dataset",
            license_id=license_info.identifier,
        )
        manifest = DatasetManifest(
            dataset_version="compliance-1.0.0",
            schema_version="1.0.0",
            created_at="2026-08-08T00:00:00Z",
            entries=(entry,),
            licenses={license_info.identifier: license_info},
            split_strategy={"seed": 42},
            frozen_test=True,
            statistics={"test": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write_atomic(path)
            self.assertEqual(DatasetManifest.load(path), manifest)

    def test_rejects_unknown_license(self) -> None:
        entry = ManifestEntry("s1", "e1", "1.png", "train", "a" * 64, "0" * 16, "owned", "missing")
        with self.assertRaisesRegex(SchemaError, "unknown licenses"):
            DatasetManifest(
                "compliance-1.0.0",
                "1.0.0",
                "2026-08-08T00:00:00Z",
                (entry,),
                {},
                {},
                False,
            )

    def test_build_manifest_computes_hashes_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            Image.new("RGB", (10, 10), "white").save(root / "images" / "000001.png")
            sample = DataSample.from_dict(sample_payload())
            manifest = build_manifest(
                [sample],
                root,
                {sample.sample_id: "test"},
                dataset_version="compliance-1.0.0",
                created_at="2026-08-08T00:00:00Z",
                frozen_test=True,
                split_strategy={"seed": 42},
            )
            self.assertEqual(manifest.statistics["test"], 1)
            self.assertEqual(len(manifest.entries[0].sha256), 64)
            self.assertEqual(len(manifest.entries[0].perceptual_hash), 16)


if __name__ == "__main__":
    unittest.main()
