"""Executable no-weight smoke check for the complete model-adapter path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_model_config
from .contracts import GenerationConfig, ModelRequest
from .mock_backend import MockBackend
from .qwen3_vl import Qwen3VLAdapter
from .serialization import model_result_to_dict

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/models/qwen3_vl_2b_mlx_4bit.json"
)


def run_mock_smoke(
    query: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    width: int = 100,
    height: int = 80,
    seed: int = 20260808,
) -> dict[str, object]:
    config = load_model_config(config_path)
    adapter = Qwen3VLAdapter(config, MockBackend())
    result = adapter.analyze(
        ModelRequest(
            image=b"mock-image-not-decoded",
            image_width=width,
            image_height=height,
            query=query,
            generation=GenerationConfig(seed=seed),
        )
    )
    return model_result_to_dict(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="找出违规区域")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args(argv)
    payload = run_mock_smoke(
        args.query,
        config_path=args.config,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
