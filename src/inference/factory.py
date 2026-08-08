"""Construction helpers for registered model configurations."""

from __future__ import annotations

from pathlib import Path

from .config import load_model_config
from .errors import UnsupportedBackendError
from .mlx_backend import MlxVlmBackend
from .qwen3_vl import Qwen3VLAdapter


def create_adapter(config_path: str | Path) -> Qwen3VLAdapter:
    config = load_model_config(config_path)
    if config.backend != "mlx_vlm":
        raise UnsupportedBackendError(f"unsupported backend: {config.backend}")
    if config.family != "qwen3_vl":
        raise UnsupportedBackendError(f"unsupported model family: {config.family}")
    return Qwen3VLAdapter(config=config, backend=MlxVlmBackend(config))
