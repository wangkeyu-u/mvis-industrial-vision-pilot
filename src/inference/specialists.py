"""Reserved specialist adapters for milestone-four collaboration work."""

from __future__ import annotations

from .base import SpecialistAdapter
from .contracts import ModelRequest, SpecialistResult
from .errors import ModelNotReadyError


class _ReservedSpecialistAdapter(SpecialistAdapter):
    implementation_name = "reserved"

    @property
    def ready(self) -> bool:
        return False

    def load(self) -> None:
        raise ModelNotReadyError(
            f"{self.implementation_name} adapter is reserved but not implemented in milestone 1"
        )

    def detect(self, request: ModelRequest) -> SpecialistResult:
        del request
        raise ModelNotReadyError(
            f"{self.implementation_name} adapter is reserved but not implemented in milestone 1"
        )


class Florence2Adapter(_ReservedSpecialistAdapter):
    implementation_name = "Florence-2"


class RFDETRAdapter(_ReservedSpecialistAdapter):
    implementation_name = "RF-DETR"
