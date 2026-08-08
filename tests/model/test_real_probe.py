import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.inference import GenerationConfig, MockBackend, Qwen3VLAdapter, load_model_config
from src.inference.real_probe import probe_adapter, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json"


class RealProbeTests(unittest.TestCase):
    def test_probe_records_real_outputs_hashes_and_latency_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "probe.png"
            Image.new("RGB", (16, 12), "white").save(image_path)
            adapter = Qwen3VLAdapter(load_model_config(CONFIG), MockBackend())

            report = probe_adapter(
                adapter,
                (image_path,),
                query="check",
                generation=GenerationConfig(max_tokens=32),
            )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["performance"]["successful_runs"], 1)
        self.assertEqual(len(report["probes"][0]["image_sha256"]), 64)
        self.assertIn("raw_text", report["probes"][0]["result"])

    def test_atomic_writer_produces_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json_atomic({"status": "recorded"}, path)

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "recorded")


if __name__ == "__main__":
    unittest.main()
