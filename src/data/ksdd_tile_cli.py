"""Export frozen KSDD tile packages for MLX-VLM and Anomalib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from .ksdd_tiles import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_TILE_SIZES,
    export_ksdd_tiles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="frozen KSDD V0 root")
    parser.add_argument("--source-lock", required=True, help="frozen KSDD V0 lock JSON")
    parser.add_argument("--output-root", required=True, help="new tiled derivative directory")
    parser.add_argument("--record-schema", required=True)
    parser.add_argument("--sft-record-schema", required=True)
    parser.add_argument("--sft-answer-schema", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=list(DEFAULT_TILE_SIZES))
    parser.add_argument(
        "--reference-root",
        default=".",
        help="base used for MLX-VLM image references",
    )
    return parser


def _source_manifest_hash(path: str | Path) -> str:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("manifest_sha256"), str):
        raise ValueError("source lock must contain manifest_sha256")
    return str(value["manifest_sha256"])


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = export_ksdd_tiles(
        arguments.source_root,
        arguments.output_root,
        expected_source_manifest_sha256=_source_manifest_hash(arguments.source_lock),
        created_at=arguments.created_at,
        record_schema_path=arguments.record_schema,
        sft_record_schema_path=arguments.sft_record_schema,
        sft_answer_schema_path=arguments.sft_answer_schema,
        dataset_version=arguments.dataset_version,
        prompt_version=arguments.prompt_version,
        tile_sizes=arguments.tile_sizes,
        reference_root=arguments.reference_root,
    )
    print(
        json.dumps(
            {
                "dataset_version": result.dataset_version,
                "source_manifest_sha256": result.source_manifest_sha256,
                "tile_manifest_sha256": result.tile_manifest_sha256,
                "tile_counts": dict(result.tile_counts),
                "leakage_passes": result.leakage_report["passes"],
                "pilot_only": result.pilot_only,
                "model_run": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
