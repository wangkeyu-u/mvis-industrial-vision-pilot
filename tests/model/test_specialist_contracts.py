import unittest

from src.inference.contracts import ModelRequest
from src.inference.errors import ModelNotReadyError
from src.inference.specialists import Florence2Adapter, RFDETRAdapter


class SpecialistContractTests(unittest.TestCase):
    def test_reserved_specialist_adapters_fail_explicitly(self) -> None:
        for adapter in (Florence2Adapter(), RFDETRAdapter()):
            with self.subTest(adapter=type(adapter).__name__):
                self.assertFalse(adapter.ready)
                with self.assertRaisesRegex(ModelNotReadyError, "reserved"):
                    adapter.detect(
                        ModelRequest(
                            image=object(), image_width=8, image_height=8, query="target"
                        )
                    )


if __name__ == "__main__":
    unittest.main()
