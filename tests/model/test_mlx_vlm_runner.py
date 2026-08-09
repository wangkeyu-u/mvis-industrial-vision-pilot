import json
import unittest
from pathlib import Path

from src.training.mlx_vlm_runner import (
    normalize_sft_record_for_adapter,
    validate_sft_package,
)

ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = ROOT / "artifacts/model/phase6/ksdd_sft_v1"


class MLXVLMRunnerTests(unittest.TestCase):
    def test_contract_transform_adds_only_null_refusal_code(self) -> None:
        record = {
            "sample_id": "one",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "inspect"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "result": "compliant",
                                    "objects": [],
                                    "reason": "none",
                                    "uncertain": False,
                                }
                            ),
                        }
                    ],
                },
            ],
        }

        normalized = normalize_sft_record_for_adapter(record)
        answer = json.loads(normalized["messages"][-1]["content"][0]["text"])

        self.assertIsNone(answer["refusal_code"])
        self.assertNotIn("refusal_code", json.loads(record["messages"][-1]["content"][0]["text"]))

    def test_contract_transform_rejects_uncertain_training_answer(self) -> None:
        record = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"uncertain": True}),
                        }
                    ],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "uncertain=false"):
            normalize_sft_record_for_adapter(record)

    @unittest.skipUnless(SFT_ROOT.is_dir(), "real SFT package is an ignored runtime artifact")
    def test_real_sft_package_hashes_and_isolation_validate(self) -> None:
        manifest = validate_sft_package(SFT_ROOT)

        self.assertEqual(manifest["statistics"]["split_counts"]["train"], 271)
        self.assertEqual(manifest["statistics"]["split_counts"]["valid"], 56)


if __name__ == "__main__":
    unittest.main()
