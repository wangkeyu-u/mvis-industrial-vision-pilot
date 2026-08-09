import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.core.schemas import AnalyzeOptions, AnalyzeTask, ImageMetadata, ObjectSource
from src.core.specialist_runtime import SpecialistAdapter, SpecialistRequest
from src.inference.service_bridge import create_specialist_service_adapter

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts/model/phase7/patchcore_resnet18/run_manifest.json"
DATASET = ROOT / "data/processed/ksdd_v0"


class SpecialistServiceBridgeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MANIFEST.is_file():
            raise unittest.SkipTest("phase 7 PatchCore runtime artifact is unavailable")
        cls.bridge = create_specialist_service_adapter(MANIFEST)

    def test_identity_ready_and_protocol(self) -> None:
        self.assertIsInstance(self.bridge, SpecialistAdapter)
        self.assertTrue(self.bridge.ready)
        self.assertEqual(self.bridge.identity.base, "anomalib/patchcore-resnet18-ksdd-v0")
        self.assertRegex(self.bridge.identity.revision, r"^[0-9a-f]{64}$")

    async def test_real_detect_returns_score_threshold_bbox_and_png(self) -> None:
        request = self._request()

        output = await self.bridge.detect(request)

        self.assertGreaterEqual(output.score, 0)
        self.assertGreater(output.threshold, 0)
        self.assertIs(output.source, ObjectSource.PATCHCORE)
        self.assertTrue((output.heatmap_png or b"").startswith(b"\x89PNG\r\n\x1a\n"))
        for item in output.objects:
            x1, y1, x2, y2 = item.bbox
            self.assertTrue(0 <= x1 < x2 <= request.image.width)
            self.assertTrue(0 <= y1 < y2 <= request.image.height)

    async def test_async_cancellation_propagates(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow_detect(request):  # type: ignore[no-untyped-def]
            del request
            started.set()
            release.wait(timeout=1)

        with patch.object(self.bridge, "_detect_sync", side_effect=slow_detect):
            task = asyncio.create_task(self.bridge.detect(self._request()))
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.001)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()

    def test_checkpoint_hash_mismatch_is_rejected_before_load(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["artifacts"]["checkpoint_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                create_specialist_service_adapter(path)

    @staticmethod
    def _request() -> SpecialistRequest:
        manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        entry = next(item for item in manifest["entries"] if item["split"] == "test")
        image_path = DATASET / entry["image"]
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = (image.format or "png").lower()
            mode = image.mode
        payload = image_path.read_bytes()
        return SpecialistRequest(
            image_bytes=payload,
            image=ImageMetadata(
                width=width,
                height=height,
                format=image_format,
                mode=mode,
                byte_size=len(payload),
            ),
            request_id="phase7-bridge-test",
            query="inspect surface anomaly",
            task=AnalyzeTask.INSPECT,
            options=AnalyzeOptions(),
        )


if __name__ == "__main__":
    unittest.main()
