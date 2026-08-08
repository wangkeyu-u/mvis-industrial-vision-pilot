"""Parameter-efficient training contracts; no trainer is executed in milestone 1."""

from .config import load_training_config
from .contracts import (
    AdaptationMethod,
    AdapterTrainingConfig,
    ModuleStrategy,
    ParameterEfficientTrainer,
    TrainingRequest,
    TrainingRun,
)
from .experiments import (
    DryRunReport,
    EvidenceStatus,
    EvidenceValue,
    ExperimentKind,
    ExperimentSpec,
    RunManifest,
    RunStatus,
    generate_default_matrix,
    validate_experiment_matrix,
)

__all__ = [
    "AdaptationMethod",
    "AdapterTrainingConfig",
    "DryRunReport",
    "EvidenceStatus",
    "EvidenceValue",
    "ExperimentKind",
    "ExperimentSpec",
    "ModuleStrategy",
    "ParameterEfficientTrainer",
    "RunManifest",
    "RunStatus",
    "TrainingRequest",
    "TrainingRun",
    "load_training_config",
    "generate_default_matrix",
    "validate_experiment_matrix",
]
