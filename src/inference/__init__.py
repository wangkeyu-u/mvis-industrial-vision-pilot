"""Model inference boundary for the multimodal compliance system."""

from .base import FusionPolicy, ModelAdapter, MultiSpecialistFusionPolicy, SpecialistAdapter
from .cache import ModelCacheStatus, inspect_model_cache
from .config import ModelConfig, load_model_config
from .contracts import (
    Decision,
    DetectedObject,
    EvidenceSource,
    GenerationConfig,
    ModelProvenance,
    ModelRequest,
    ModelResult,
    RefusalCode,
    RefusalSignal,
    Task,
)
from .export import BatchPredictionSample, PredictionExportSummary, export_predictions_jsonl
from .factory import create_adapter
from .fusion import ConservativeFusionPolicy
from .mock_backend import MockBackend
from .performance import (
    PerformanceSample,
    PerformanceSampler,
    PerformanceSummary,
    unavailable_performance,
)
from .qwen3_vl import Qwen3VLAdapter
from .serialization import model_result_to_dict

__all__ = [
    "BatchPredictionSample",
    "ConservativeFusionPolicy",
    "Decision",
    "DetectedObject",
    "EvidenceSource",
    "FusionPolicy",
    "GenerationConfig",
    "MockBackend",
    "ModelCacheStatus",
    "ModelAdapter",
    "ModelConfig",
    "ModelProvenance",
    "ModelRequest",
    "ModelResult",
    "MultiSpecialistFusionPolicy",
    "PerformanceSample",
    "PerformanceSampler",
    "PerformanceSummary",
    "PredictionExportSummary",
    "Qwen3VLAdapter",
    "RefusalCode",
    "RefusalSignal",
    "SpecialistAdapter",
    "Task",
    "create_adapter",
    "export_predictions_jsonl",
    "inspect_model_cache",
    "load_model_config",
    "model_result_to_dict",
    "unavailable_performance",
]
