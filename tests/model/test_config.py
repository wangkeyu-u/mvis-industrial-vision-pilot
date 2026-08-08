import json
import tempfile
import unittest
from pathlib import Path

from src.inference import GenerationConfig, load_model_config
from src.training import AdaptationMethod, load_training_config

ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_mlx_4bit_config_is_deterministic_and_offline_by_default(self) -> None:
        config = load_model_config(ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json")

        self.assertEqual(config.family, "qwen3_vl")
        self.assertEqual(config.backend, "mlx_vlm")
        self.assertEqual(config.quantization.bits, 4)
        self.assertTrue(config.generation.deterministic)
        self.assertFalse(config.allow_download)
        self.assertNotEqual(config.revision, "main")
        self.assertEqual(config.generation.max_tokens, 512)
        self.assertEqual(len(config.fingerprint), 16)

    def test_deterministic_generation_rejects_nonzero_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            GenerationConfig(do_sample=False, temperature=0.2)

    def test_qlora_probe_config_targets_4bit_batch_one(self) -> None:
        config = load_training_config(
            ROOT / "configs/models/qwen3_vl_2b_qlora_m5_16gb.json"
        )

        self.assertIs(config.method, AdaptationMethod.QLORA)
        self.assertEqual(config.quantization_bits, 4)
        self.assertEqual(config.batch_size, 1)
        self.assertTrue(config.gradient_checkpointing)

    def test_model_config_rejects_unknown_and_string_boolean(self) -> None:
        source = json.loads(
            (ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json").read_text(
                encoding="utf-8"
            )
        )
        for field, value, message in (
            ("allow_download", "false", "must be a boolean"),
            ("misspelled_option", 1, "unknown model config fields"),
        ):
            candidate = dict(source)
            candidate[field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "model.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_model_config(path)

        candidate = dict(source)
        candidate["generation"] = dict(source["generation"])
        candidate["generation"]["do_sample"] = "false"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generation.do_sample"):
                load_model_config(path)

    def test_training_config_rejects_string_integer(self) -> None:
        source = json.loads(
            (ROOT / "configs/models/qwen3_vl_2b_qlora_m5_16gb.json").read_text(
                encoding="utf-8"
            )
        )
        source["rank"] = "8"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rank must be an integer"):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()
