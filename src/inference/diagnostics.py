"""Read-only model runtime diagnostics; this command never downloads weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from .cache import inspect_model_cache
from .config import ModelConfig, load_model_config
from .performance import unavailable_performance

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/models/qwen3_vl_2b_mlx_4bit.json"
)

DEPENDENCIES = {
    "mlx": "mlx",
    "mlx_vlm": "mlx-vlm",
    "huggingface_hub": "huggingface-hub",
    "pydantic": "pydantic",
    "pillow": "Pillow",
    "fastapi": "fastapi",
}


def collect_dependency_status() -> dict[str, dict[str, str | bool | None]]:
    status: dict[str, dict[str, str | bool | None]] = {}
    for label, distribution in DEPENDENCIES.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        status[label] = {
            "distribution": distribution,
            "installed": version is not None,
            "version": version,
        }
    return status


def diagnose_model(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_model_config(config_path)
    dependencies = collect_dependency_status()
    cache = inspect_model_cache(config, cache_root=cache_root)
    reasons = _readiness_reasons(config, dependencies, cache.complete)
    ready = not reasons
    performance_reason = (
        "real_model_unavailable: " + "; ".join(reasons)
        if reasons
        else "not_sampled: diagnostics is read-only and no probe image was supplied"
    )
    return {
        "schema_version": "1.0",
        "status": "ready" if ready else "unavailable",
        "ready": ready,
        "reasons": reasons,
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "total_memory_bytes": _total_memory_bytes(),
        },
        "dependencies": dependencies,
        "model": {
            "alias": config.alias,
            "model_id": config.model_id,
            "base_model_id": config.base_model_id,
            "revision": config.revision,
            "backend": config.backend,
            "adapter_path": config.adapter_path,
            "config_fingerprint": config.fingerprint,
            "allow_download": config.allow_download,
            "trust_remote_code": config.trust_remote_code,
        },
        "quantization": {
            "bits": config.quantization.bits,
            "group_size": config.quantization.group_size,
            "mode": config.quantization.mode,
        },
        "generation": {
            "seed": config.generation.seed,
            "do_sample": config.generation.do_sample,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "max_tokens": config.generation.max_tokens,
            "deterministic": config.generation.deterministic,
        },
        "limits": dict(config.limits),
        "cache": cache.as_dict(),
        "performance": unavailable_performance(performance_reason).as_dict(),
    }


def _readiness_reasons(
    config: ModelConfig,
    dependencies: dict[str, dict[str, str | bool | None]],
    cache_complete: bool,
) -> list[str]:
    reasons: list[str] = []
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        reasons.append("mlx_requires_apple_silicon")
    for dependency in ("mlx", "mlx_vlm"):
        if not dependencies[dependency]["installed"]:
            reasons.append(f"dependency_missing:{dependency}")
    if not cache_complete:
        reasons.append("pinned_model_snapshot_unavailable")
    if config.trust_remote_code:
        reasons.append("trust_remote_code_must_be_false")
    return reasons


def _total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(page_count, int):
        return None
    return page_size * page_count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    payload = diagnose_model(args.config, cache_root=args.cache_root)
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=None if args.compact else 2,
        )
    )
    return 2 if args.require_ready and not payload["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
