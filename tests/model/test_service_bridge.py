import asyncio
import io
import json
import time
import unittest
from pathlib import Path

from PIL import Image

from src.core.analyze_service import AnalyzeCommand, AnalyzeService
from src.core.config import ServiceSettings
from src.core.errors import ErrorCode, ServiceError
from src.core.model_registry import (
    AdapterRequest,
    ModelRegistration,
    ModelRegistry,
    ModelState,
)
from src.core.model_registry import (
    ModelAdapter as ServiceModelAdapter,
)
from src.core.schemas import AnalyzeOptions, AnalyzeTask, ImageMetadata
from src.inference import MockBackend
from src.inference.base import GenerationBackend
from src.inference.contracts import BackendRequest, BackendResponse
from src.inference.service_bridge import create_service_adapter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json"


def adapter_request(query: str = "找出违规区域") -> AdapterRequest:
    return AdapterRequest(
        image_bytes=b"mock",
        image=ImageMetadata(
            width=100, height=80, format="jpeg", mode="RGB", byte_size=4
        ),
        query=query,
        task=AnalyzeTask.INSPECT,
        use_specialist=False,
        options=AnalyzeOptions(),
    )


def jpeg_bytes(size: tuple[int, int] = (100, 80)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(output, format="JPEG")
    return output.getvalue()


def registry_for(backend: GenerationBackend) -> tuple[ModelRegistry, object]:
    bridge = create_service_adapter(CONFIG, backend=backend)
    registry = ModelRegistry()
    registry.register(
        ModelRegistration(
            model_id="qwen3-vl-2b-instruct-4bit",
            adapter=bridge,
            state=ModelState.ACTIVE,
            source="model-test",
            quantization="4-bit",
        )
    )
    registry.set_runtime_status(
        requested_mode="qwen3-vl-mlx",
        selected_mode="model-test",
        degraded=False,
    )
    return registry, bridge


def analyze_command() -> AnalyzeCommand:
    return AnalyzeCommand(
        image=jpeg_bytes(),
        image_content_type="image/jpeg",
        query="找出违规区域",
        task=AnalyzeTask.INSPECT,
        model="active",
        use_specialist=False,
        options=AnalyzeOptions(),
    )


class SlowBackend:
    def __init__(self, delay_seconds: float = 0.05) -> None:
        self.delay_seconds = delay_seconds

    @property
    def name(self) -> str:
        return "slow-test"

    @property
    def ready(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def generate(self, request: BackendRequest) -> BackendResponse:
        del request
        time.sleep(self.delay_seconds)
        return BackendResponse(
            text=json.dumps(
                {
                    "result": "compliant",
                    "objects": [],
                    "reason": "slow deterministic response",
                    "refusal_code": None,
                }
            )
        )


class ServiceBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_adapter_is_directly_registry_compatible(self) -> None:
        registry, bridge = registry_for(MockBackend())

        self.assertIsInstance(bridge, ServiceModelAdapter)
        self.assertIs(registry.resolve("active").adapter, bridge)
        self.assertEqual(registry.runtime_status()["selected_mode"], "model-test")
        output = await bridge.analyze(adapter_request())

        self.assertEqual(output.result, "violation")
        self.assertFalse(output.uncertain)
        self.assertEqual(output.objects[0].source.value, "vlm")
        self.assertEqual(bridge.identity.base, "qwen3-vl-2b-instruct-4bit")

    async def test_uncertain_refusal_survives_service_contract(self) -> None:
        _, bridge = registry_for(MockBackend())

        output = await bridge.analyze(adapter_request("看不清，证据不足"))

        self.assertEqual(output.result, "uncertain")
        self.assertTrue(output.uncertain)
        self.assertEqual(output.objects, [])
        self.assertIn("refusal:insufficient_evidence", output.warnings)

    async def test_invalid_json_maps_to_output_validation_failed(self) -> None:
        registry, _ = registry_for(MockBackend(lambda _: "not-json"))
        service = AnalyzeService(ServiceSettings(config_path="test"), registry)

        with self.assertRaises(ServiceError) as raised:
            await service.analyze(analyze_command(), "req_invalid_json")

        self.assertIs(raised.exception.code, ErrorCode.OUTPUT_VALIDATION_FAILED)
        self.assertEqual(raised.exception.status_code, 422)

    async def test_out_of_bounds_bbox_maps_to_output_validation_failed(self) -> None:
        payload = {
            "result": "violation",
            "objects": [
                {"label": "target", "bbox": [1, 1, 101, 20], "confidence": 0.8}
            ],
            "reason": "out of bounds",
            "refusal_code": None,
        }
        registry, _ = registry_for(
            MockBackend(lambda _: json.dumps(payload, ensure_ascii=False))
        )
        service = AnalyzeService(ServiceSettings(config_path="test"), registry)

        with self.assertRaises(ServiceError) as raised:
            await service.analyze(analyze_command(), "req_bbox")

        self.assertIs(raised.exception.code, ErrorCode.OUTPUT_VALIDATION_FAILED)
        self.assertEqual(raised.exception.status_code, 422)

    async def test_timeout_is_mapped_by_backend_wait_for(self) -> None:
        registry, _ = registry_for(SlowBackend())
        settings = ServiceSettings(
            config_path="test",
            inference_timeout_seconds=0.001,
            concurrency_wait_seconds=0.01,
        )
        service = AnalyzeService(settings, registry)

        with self.assertRaises(ServiceError) as raised:
            await service.analyze(analyze_command(), "req_timeout")

        self.assertIs(raised.exception.code, ErrorCode.INFERENCE_TIMEOUT)
        self.assertEqual(raised.exception.status_code, 504)

    async def test_cancellation_is_propagated_not_converted(self) -> None:
        _, bridge = registry_for(SlowBackend())
        task = asyncio.create_task(bridge.analyze(adapter_request()))
        await asyncio.sleep(0)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    unittest.main()
