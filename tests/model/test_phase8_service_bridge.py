"""Phase 8 specialist bridge dispatch and validation contract tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.core.schemas import AnalyzeOptions, AnalyzeTask, ImageMetadata, ObjectSource
from src.core.specialist_runtime import SpecialistAdapter, SpecialistRequest
from src.inference.service_bridge import (
    PatchcoreServiceAdapterBridge,
    UnetServiceAdapterBridge,
    create_specialist_service_adapter,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE7_MANIFEST = ROOT / "artifacts/model/phase7/patchcore_resnet18/run_manifest.json"
PHASE8_MANIFEST = ROOT / "artifacts/model/phase8/specialist_manifest_unet.json"
DATASET = ROOT / "data/processed/ksdd_v0"


def _minimal_unet_manifest(**overrides: object) -> dict:
    manifest = {
        "schema_version": "1.0",
        "phase": "phase8",
        "kind": "phase8_specialist_manifest",
        "algorithm": "unet",
        "status": "completed",
        "test_labels_used_for_selection": False,
        "checkpoint": "/nonexistent/unet.pt",
        "checkpoint_sha256": "0" * 64,
        "dataset_manifest_sha256": "f" * 64,
        "image_threshold": 0.9,
        "fusion_mode": "mean",
        "postprocess_config": {
            "name": "test",
            "threshold_mode": "quantile",
            "threshold": 0.95,
            "max_components": 3,
            "min_area": 8,
            "morphology": "none",
            "merge_vertical_gap": 0,
            "thin_aspect": 0.0,
        },
        "input_size": 256,
        "tile_size": 256,
        "tile_stride": 128,
        "device": "mps",
        "encoder": {
            "repository": "timm/resnet18.a1_in1k",
            "revision": "r" * 40,
            "sha256": "e" * 64,
            "bytes": 46807446,
        },
    }
    manifest.update(overrides)
    return manifest


class UnetManifestValidationTests(unittest.TestCase):
    def _write_manifest(self, payload: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        with handle:
            json.dump(payload, handle)
        return Path(handle.name)

    def test_rejects_missing_checkpoint(self) -> None:
        path = self._write_manifest(_minimal_unet_manifest())
        with self.assertRaises(ValueError):
            create_specialist_service_adapter(path)
        path.unlink()

    def test_rejects_test_label_selection(self) -> None:
        path = self._write_manifest(
            _minimal_unet_manifest(test_labels_used_for_selection=True)
        )
        with self.assertRaises(ValueError):
            create_specialist_service_adapter(path)
        path.unlink()

    def test_rejects_bad_fusion_mode(self) -> None:
        path = self._write_manifest(_minimal_unet_manifest(fusion_mode="median"))
        with self.assertRaises(ValueError):
            create_specialist_service_adapter(path)
        path.unlink()

    def test_rejects_bad_geometry(self) -> None:
        path = self._write_manifest(_minimal_unet_manifest(tile_stride=0))
        with self.assertRaises(ValueError):
            create_specialist_service_adapter(path)
        path.unlink()

    def test_rejects_invalid_postprocess(self) -> None:
        payload = _minimal_unet_manifest()
        payload["postprocess_config"]["threshold"] = 1.5
        path = self._write_manifest(payload)
        with self.assertRaises(ValueError):
            create_specialist_service_adapter(path)
        path.unlink()

    def test_patchcore_manifest_still_dispatches_to_patchcore_loader(self) -> None:
        payload = {
            "schema_version": "1.0",
            "status": "completed",
            "implementation": {"algorithm": "patchcore"},
            "artifacts": {},
            "calibration": {},
            "configuration": {},
        }
        path = self._write_manifest(payload)
        with self.assertRaises(ValueError) as context:
            create_specialist_service_adapter(path)
        self.assertIn("checkpoint", str(context.exception))
        path.unlink()


class UnetBridgeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not PHASE8_MANIFEST.is_file():
            raise unittest.SkipTest("phase 8 U-Net specialist manifest is unavailable")
        cls.bridge = create_specialist_service_adapter(PHASE8_MANIFEST)

    def test_dispatch_returns_unet_bridge(self) -> None:
        self.assertIsInstance(self.bridge, UnetServiceAdapterBridge)
        self.assertIsInstance(self.bridge, SpecialistAdapter)
        self.assertTrue(self.bridge.ready)
        self.assertEqual(self.bridge.identity.base, "mvis/unet-resnet18-ksdd-v1")

    async def test_real_detect_outputs_native_heatmap_and_boxes(self) -> None:
        sample = DATASET / "images/kos10/Part0.jpg"
        if not sample.is_file():
            self.skipTest("KSDD sample image unavailable")
        image_bytes = sample.read_bytes()
        request = SpecialistRequest(
            image_bytes=image_bytes,
            image=ImageMetadata(
                width=500, height=1265, format="jpeg", mode="L", byte_size=len(image_bytes)
            ),
            request_id="req_phase8_test",
            query="inspect",
            task=AnalyzeTask.INSPECT,
            options=AnalyzeOptions(),
        )
        output = await self.bridge.detect(request)
        self.assertIs(output.source, ObjectSource.UNET)
        self.assertGreaterEqual(output.score, 0.0)
        self.assertGreater(output.threshold, 0.0)
        self.assertTrue((output.heatmap_png or b"").startswith(b"\x89PNG\r\n\x1a\n"))
        for item in output.objects:
            x1, y1, x2, y2 = item.bbox
            self.assertTrue(0 <= x1 < x2 <= request.image.width)
            self.assertTrue(0 <= y1 < y2 <= request.image.height)
            self.assertIs(item.source, ObjectSource.UNET)


class PatchcoreRollbackTests(unittest.TestCase):
    def test_phase7_manifest_remains_loadable(self) -> None:
        if not PHASE7_MANIFEST.is_file():
            self.skipTest("phase 7 PatchCore artifact unavailable")
        bridge = create_specialist_service_adapter(PHASE7_MANIFEST)
        self.assertIsInstance(bridge, PatchcoreServiceAdapterBridge)
        self.assertTrue(bridge.ready)


if __name__ == "__main__":
    unittest.main()
