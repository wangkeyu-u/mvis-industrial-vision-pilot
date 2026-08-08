import unittest

from src.inference.smoke import run_mock_smoke


class SmokeTests(unittest.TestCase):
    def test_smoke_returns_serializable_violation(self) -> None:
        first = run_mock_smoke("找出违规区域", seed=17)
        second = run_mock_smoke("找出违规区域", seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first["result"], "violation")
        self.assertEqual(first["objects"][0]["source"], "vlm")
        self.assertEqual(first["provenance"]["seed"], 17)
        self.assertTrue(first["provenance"]["deterministic"])

    def test_smoke_serializes_refusal(self) -> None:
        payload = run_mock_smoke("看不清，证据不足")

        self.assertEqual(payload["result"], "uncertain")
        self.assertEqual(payload["refusal"]["code"], "insufficient_evidence")
        self.assertTrue(payload["refusal"]["review_required"])


if __name__ == "__main__":
    unittest.main()
