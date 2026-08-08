"""Framework-neutral contracts reserved for LoRA/QLoRA training."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol


class AdaptationMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"


class ModuleStrategy(str, Enum):
    LANGUAGE_ONLY = "language_only"
    VISION_LANGUAGE = "vision_language"


@dataclass(frozen=True, slots=True)
class AdapterTrainingConfig:
    base_model_id: str
    method: AdaptationMethod
    module_strategy: ModuleStrategy
    rank: int
    alpha: int
    dropout: float
    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    gradient_checkpointing: bool = True
    quantization_bits: int | None = 4
    extra: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_model_id.strip():
            raise ValueError("base_model_id is required")
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if min(self.batch_size, self.gradient_accumulation_steps, self.max_steps) <= 0:
            raise ValueError("batch and step values must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.method is AdaptationMethod.QLORA and self.quantization_bits != 4:
            raise ValueError("milestone QLoRA contract requires 4-bit base weights")


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    run_id: str
    data_manifest: Path
    output_dir: Path
    config: AdapterTrainingConfig

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")


@dataclass(frozen=True, slots=True)
class TrainingRun:
    run_id: str
    status: str
    adapter_path: Path | None
    metrics: Mapping[str, float]
    metadata: Mapping[str, str]


class ParameterEfficientTrainer(Protocol):
    """Implementation seam for a future MLX or remote-GPU trainer."""

    def train(self, request: TrainingRequest) -> TrainingRun: ...
