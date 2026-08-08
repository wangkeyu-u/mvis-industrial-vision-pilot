import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.inference import create_adapter
from src.inference.errors import ModelNotReadyError

ROOT = Path(__file__).resolve().parents[2]


class MlxBackendTests(unittest.TestCase):
    def test_construction_does_not_import_or_download_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"HF_HOME": directory}
        ):
            adapter = create_adapter(ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json")

            self.assertFalse(adapter.ready)
            with self.assertRaisesRegex(ModelNotReadyError, "download is disabled"):
                adapter.load()


if __name__ == "__main__":
    unittest.main()
