"""CLI for exporting frozen KSDD SFT V1 in the official MLX-VLM message format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from .ksdd_sft import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_PROMPT_VERSION,
    export_ksdd_sft,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="frozen KSDD V0 root")
    parser.add_argument("--source-lock", required=True, help="KSDD V0 lock JSON")
    parser.add_argument("--output-root", required=True, help="new SFT V1 output directory")
    parser.add_argument("--record-schema", required=True)
    parser.add_argument("--answer-schema", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument(
        "--reference-root",
        default=".",
        help="base used to make image references relative; run training from this directory",
    )
    return parser


def _load_source_lock(path: str | Path) -> Mapping[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("source lock must be a JSON object")
    manifest_hash = value.get("manifest_sha256")
    if not isinstance(manifest_hash, str):
        raise ValueError("source lock manifest_sha256 is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    source_lock = _load_source_lock(arguments.source_lock)
    result = export_ksdd_sft(
        arguments.source_root,
        arguments.output_root,
        expected_source_manifest_sha256=str(source_lock["manifest_sha256"]),
        created_at=arguments.created_at,
        record_schema_path=arguments.record_schema,
        answer_schema_path=arguments.answer_schema,
        dataset_version=arguments.dataset_version,
        prompt_version=arguments.prompt_version,
        reference_root=arguments.reference_root,
    )
    print(
        json.dumps(
            {
                "dataset_version": result.dataset_version,
                "prompt_version": result.prompt_version,
                "source_manifest_sha256": result.source_manifest_sha256,
                "sft_manifest_sha256": result.manifest_sha256,
                "split_counts": dict(result.split_counts),
                "leakage_passes": result.leakage_report["passes"],
                "pilot_only": True,
                "model_run": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
