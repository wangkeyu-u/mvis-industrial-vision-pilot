"""Abstract model and specialist adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence

from .contracts import BackendRequest, BackendResponse, ModelRequest, ModelResult, SpecialistResult


class GenerationBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def ready(self) -> bool: ...

    def load(self) -> None: ...

    def generate(self, request: BackendRequest) -> BackendResponse: ...


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def analyze(self, request: ModelRequest) -> ModelResult:
        raise NotImplementedError


class SpecialistAdapter(ABC):
    """Common seam for Florence-2 and RF-DETR implementations."""

    @property
    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def detect(self, request: ModelRequest) -> SpecialistResult:
        raise NotImplementedError


class FusionPolicy(Protocol):
    """Policy seam; orchestration may combine VLM and specialist evidence."""

    def fuse(self, primary: ModelResult, specialist: SpecialistResult) -> ModelResult: ...


class MultiSpecialistFusionPolicy(FusionPolicy, Protocol):
    """Optional extension for jointly evaluating multiple specialist sources."""

    def fuse_many(
        self, primary: ModelResult, specialists: Sequence[SpecialistResult]
    ) -> ModelResult: ...
