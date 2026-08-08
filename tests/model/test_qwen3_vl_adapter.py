import json
import unittest
from pathlib import Path

from src.inference import (
    Decision,
    GenerationConfig,
    MockBackend,
    ModelRequest,
    Qwen3VLAdapter,
    RefusalCode,
    load_model_config,
)
from src.inference.errors import ModelOutputError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json"


def adapter_with_response(payload: dict) -> Qwen3VLAdapter:
    backend = MockBackend(lambda _: json.dumps(payload, ensure_ascii=False))
    return Qwen3VLAdapter(load_model_config(CONFIG), backend)


class Qwen3VLAdapterTests(unittest.TestCase):
    def test_mock_inference_is_repeatable_with_fixed_seed(self) -> None:
        backend = MockBackend()
        adapter = Qwen3VLAdapter(load_model_config(CONFIG), backend)
        request = ModelRequest(
            image=b"not-decoded-by-model-layer",
            image_width=100,
            image_height=80,
            query="找出违规区域",
            generation=GenerationConfig(seed=7),
        )

        first = adapter.analyze(request)
        second = adapter.analyze(request)

        self.assertEqual(first, second)
        self.assertIs(first.decision, Decision.VIOLATION)
        self.assertEqual(first.objects[0].bbox, (10, 12, 40, 44))
        self.assertTrue(first.provenance.deterministic)
        self.assertEqual(first.provenance.seed, 7)
        self.assertEqual(backend.calls[0].prompt, backend.calls[1].prompt)

    def test_insufficient_evidence_returns_explicit_refusal(self) -> None:
        backend = MockBackend()
        adapter = Qwen3VLAdapter(load_model_config(CONFIG), backend)

        result = adapter.analyze(
            ModelRequest(
                image=object(),
                image_width=100,
                image_height=80,
                query="图像模糊，看不清，证据不足",
            )
        )

        self.assertIs(result.decision, Decision.UNCERTAIN)
        self.assertTrue(result.uncertain)
        self.assertEqual(result.objects, ())
        self.assertIsNotNone(result.refusal)
        assert result.refusal is not None
        self.assertIs(result.refusal.code, RefusalCode.INSUFFICIENT_EVIDENCE)
        self.assertTrue(result.refusal.review_required)

    def test_compliant_result_cannot_contain_positive_evidence(self) -> None:
        adapter = adapter_with_response(
            {
                "result": "compliant",
                "objects": [{"label": "x", "bbox": [1, 1, 2, 2], "confidence": 0.8}],
                "reason": "contradiction",
                "refusal_code": None,
            }
        )

        with self.assertRaisesRegex(ModelOutputError, "contradictory"):
            adapter.analyze(
                ModelRequest(image=object(), image_width=10, image_height=10, query="check")
            )

    def test_bbox_must_be_integer_original_pixel_coordinates(self) -> None:
        adapter = adapter_with_response(
            {
                "result": "violation",
                "objects": [{"label": "x", "bbox": [1, 1, 11, 2], "confidence": 0.8}],
                "reason": "outside",
                "refusal_code": None,
            }
        )

        with self.assertRaisesRegex(ModelOutputError, "bounds"):
            adapter.analyze(
                ModelRequest(image=object(), image_width=10, image_height=10, query="check")
            )

    def test_malformed_model_text_is_a_stable_output_error(self) -> None:
        config = load_model_config(CONFIG)
        adapter = Qwen3VLAdapter(config, MockBackend(lambda _: "not-json"))

        with self.assertRaisesRegex(ModelOutputError, "valid JSON"):
            adapter.analyze(
                ModelRequest(image=object(), image_width=10, image_height=10, query="check")
            )

    def test_adapter_rejects_configured_image_and_token_overflow(self) -> None:
        backend = MockBackend()
        adapter = Qwen3VLAdapter(load_model_config(CONFIG), backend)

        with self.assertRaisesRegex(ValueError, "pixel limit"):
            adapter.analyze(
                ModelRequest(image=object(), image_width=5000, image_height=5000, query="check")
            )
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            adapter.analyze(
                ModelRequest(
                    image=object(),
                    image_width=10,
                    image_height=10,
                    query="check",
                    generation=GenerationConfig(max_tokens=513),
                )
            )

    def test_specialist_request_is_not_silently_ignored(self) -> None:
        adapter = Qwen3VLAdapter(load_model_config(CONFIG), MockBackend())

        result = adapter.analyze(
            ModelRequest(
                image=object(),
                image_width=100,
                image_height=80,
                query="check",
                use_specialist=True,
            )
        )

        self.assertIn("specialist_requested_but_unavailable", result.warnings)


if __name__ == "__main__":
    unittest.main()
