import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.inference import inspect_model_cache, load_model_config
from src.inference.diagnostics import diagnose_model, main

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json"


class DiagnosticsTests(unittest.TestCase):
    def test_missing_snapshot_marks_real_runtime_unavailable_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = diagnose_model(CONFIG, cache_root=temp_dir)

        self.assertFalse(payload["ready"])
        self.assertEqual(payload["status"], "unavailable")
        self.assertIn("pinned_model_snapshot_unavailable", payload["reasons"])
        self.assertEqual(payload["model"]["revision"], "9c4f5209e57b31f4b9dfba735de3fb983739c9cc")
        self.assertEqual(payload["quantization"]["bits"], 4)
        self.assertFalse(payload["performance"]["available"])
        self.assertIsNone(payload["performance"]["p50_latency_ms"])
        self.assertIsNone(payload["performance"]["p95_latency_ms"])
        self.assertIsNone(payload["performance"]["peak_memory_mb"])
        self.assertEqual(payload["performance"]["samples"], [])

    def test_cache_requires_config_and_safetensors_at_pinned_revision(self) -> None:
        config = load_model_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = (
                Path(temp_dir)
                / "models--mlx-community--Qwen3-VL-2B-Instruct-4bit"
                / "snapshots"
                / config.revision
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            incomplete = inspect_model_cache(config, cache_root=temp_dir)
            (snapshot / "model.safetensors").write_bytes(b"test-placeholder")
            complete = inspect_model_cache(config, cache_root=temp_dir)

        self.assertFalse(incomplete.complete)
        self.assertEqual(incomplete.reason, "safetensors_weights_missing")
        self.assertTrue(complete.complete)
        self.assertEqual(complete.reason, None)
        self.assertEqual(complete.weight_files, ("model.safetensors",))

    def test_cli_emits_valid_json_and_require_ready_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--config",
                        str(CONFIG),
                        "--cache-root",
                        temp_dir,
                        "--compact",
                        "--require-ready",
                    ]
                )
        payload = json.loads(output.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "unavailable")
        self.assertIn("environment", payload)
        self.assertIn("dependencies", payload)


if __name__ == "__main__":
    unittest.main()
