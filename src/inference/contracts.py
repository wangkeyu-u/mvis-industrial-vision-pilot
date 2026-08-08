"""Stable contracts shared by model adapters and their callers.

The types in this module intentionally do not expose MLX, Transformers, Qwen,
Florence, or RF-DETR response objects.  Backend-specific values stop at the
adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class Task(str, Enum):
    INSPECT = "inspect"
    GROUND = "ground"
    EXTRACT = "extract"
    VQA = "vqa"


class Decision(str, Enum):
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    UNCERTAIN = "uncertain"


class EvidenceSource(str, Enum):
    VLM = "vlm"
    FLORENCE2 = "florence2"
    RF_DETR = "rf_detr"
    FUSION = "fusion"


class RefusalCode(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_CONFLICT = "model_conflict"
    UNSUPPORTED_TASK = "unsupported_task"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generation controls with deterministic inference as the safe default."""

    seed: int = 20260808
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 1 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens must be in [1, 4096]")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not self.do_sample and self.temperature != 0.0:
            raise ValueError("deterministic generation requires temperature=0")

    @property
    def deterministic(self) -> bool:
        return not self.do_sample and self.temperature == 0.0


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One image/query request at the model boundary.

    ``image`` is deliberately opaque.  A concrete backend documents the forms
    it accepts; the MLX backend currently accepts a local path.  Image decoding,
    MIME validation, and resizing belong to the data/input layer.
    """

    image: Any
    image_width: int
    image_height: int
    query: str
    task: Task = Task.INSPECT
    request_id: str = ""
    use_specialist: bool = False
    generation: GenerationConfig | None = None

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        query = self.query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if len(query) > 1000:
            raise ValueError("query must not exceed 1000 characters")


@dataclass(frozen=True, slots=True)
class DetectedObject:
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    source: EvidenceSource = EvidenceSource.VLM

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("object label must not be empty")
        x1, y1, x2, y2 = self.bbox
        if min(self.bbox) < 0 or x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy 0 <= x1 < x2 and 0 <= y1 < y2")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RefusalSignal:
    code: RefusalCode
    message: str
    review_required: bool = True


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    model_id: str
    model_revision: str
    backend: str
    config_fingerprint: str
    adapter_id: str | None = None
    seed: int = 20260808
    deterministic: bool = True
    extra: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResult:
    decision: Decision
    objects: tuple[DetectedObject, ...]
    reason: str
    uncertain: bool
    refusal: RefusalSignal | None
    provenance: ModelProvenance
    warnings: tuple[str, ...] = ()
    raw_text: str | None = None

    def __post_init__(self) -> None:
        if self.decision is Decision.VIOLATION and not self.objects:
            raise ValueError("violation requires at least one evidence object")
        if self.decision is Decision.COMPLIANT and self.objects:
            raise ValueError("compliant result cannot contain positive evidence")
        if self.uncertain != (self.decision is Decision.UNCERTAIN):
            raise ValueError("uncertain flag must match uncertain decision")
        if self.uncertain != (self.refusal is not None):
            raise ValueError("uncertain result must carry exactly one refusal signal")


@dataclass(frozen=True, slots=True)
class BackendRequest:
    image: Any
    prompt: str
    generation: GenerationConfig
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class BackendResponse:
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpecialistResult:
    source: EvidenceSource
    objects: tuple[DetectedObject, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)


ImagePath = str | Path
ObjectSequence = Sequence[DetectedObject]
