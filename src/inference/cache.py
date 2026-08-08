"""Read-only discovery of local Hugging Face/MLX model snapshots."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig


@dataclass(frozen=True, slots=True)
class ModelCacheStatus:
    source: str
    path: str | None
    exists: bool
    complete: bool
    config_present: bool
    weight_files: tuple[str, ...]
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weight_files"] = list(self.weight_files)
        return payload


def default_huggingface_cache_root() -> Path:
    explicit = os.getenv("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache/huggingface/hub"


def inspect_model_cache(
    config: ModelConfig, *, cache_root: str | Path | None = None
) -> ModelCacheStatus:
    if config.local_model_path:
        path = Path(config.local_model_path).expanduser()
        return _inspect_snapshot(path, source="configured_local_path")

    root = (
        Path(cache_root).expanduser()
        if cache_root is not None
        else default_huggingface_cache_root()
    )
    repository_dir = "models--" + config.model_id.replace("/", "--")
    snapshot = root / repository_dir / "snapshots" / config.revision
    return _inspect_snapshot(snapshot, source="huggingface_cache")


def resolve_cached_model_path(
    config: ModelConfig, *, cache_root: str | Path | None = None
) -> Path | None:
    status = inspect_model_cache(config, cache_root=cache_root)
    return Path(status.path) if status.complete and status.path else None


def _inspect_snapshot(path: Path, *, source: str) -> ModelCacheStatus:
    exists = path.is_dir()
    config_present = (path / "config.json").is_file() if exists else False
    weights = (
        tuple(sorted(str(item.name) for item in path.glob("*.safetensors") if item.is_file()))
        if exists
        else ()
    )
    complete = exists and config_present and bool(weights)
    if complete:
        reason = None
    elif not exists:
        reason = "snapshot_directory_missing"
    elif not config_present:
        reason = "config_json_missing"
    else:
        reason = "safetensors_weights_missing"
    return ModelCacheStatus(
        source=source,
        path=str(path),
        exists=exists,
        complete=complete,
        config_present=config_present,
        weight_files=weights,
        reason=reason,
    )
