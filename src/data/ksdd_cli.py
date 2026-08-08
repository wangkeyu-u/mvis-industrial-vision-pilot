"""CLI for freezing the official KolektorSDD fine-annotation release as V0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .deduplication import sha256_file
from .ksdd import (
    KSDD_ARCHIVE_URL,
    KSDD_SOURCE_PAGE,
    KSDD_V0_ARCHIVE_BYTES,
    KSDD_V0_ARCHIVE_SHA256,
    KSDDSourceRecord,
    prepare_ksdd_dataset,
)
from .license_policy import LicensePolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the official KolektorSDD fine annotations to canonical V0"
    )
    parser.add_argument("--source-root", required=True, help="extracted directory containing kosNN")
    parser.add_argument("--archive", required=True, help="downloaded official ZIP for provenance")
    parser.add_argument("--output-root", required=True, help="new, empty frozen dataset directory")
    parser.add_argument("--license-policy", required=True)
    parser.add_argument("--downloaded-at", required=True, help="ISO-8601 download timestamp")
    parser.add_argument("--created-at", required=True, help="ISO-8601 manifest timestamp")
    parser.add_argument("--dataset-version", default="ksdd-0.1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hash-distance", type=int, default=5)
    parser.add_argument("--minimum-formal-kpi-test-samples", type=int, default=300)
    parser.add_argument("--source-page", default=KSDD_SOURCE_PAGE)
    parser.add_argument("--archive-url", default=KSDD_ARCHIVE_URL)
    parser.add_argument("--expected-archive-sha256", default=KSDD_V0_ARCHIVE_SHA256)
    parser.add_argument("--expected-archive-bytes", type=int, default=KSDD_V0_ARCHIVE_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    archive = Path(arguments.archive)
    if not archive.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    actual_size = archive.stat().st_size
    actual_hash = sha256_file(archive)
    if actual_size != arguments.expected_archive_bytes:
        raise ValueError(
            f"archive size mismatch: expected {arguments.expected_archive_bytes}, got {actual_size}"
        )
    if actual_hash != arguments.expected_archive_sha256:
        raise ValueError(
            f"archive hash mismatch: expected {arguments.expected_archive_sha256}, got {actual_hash}"
        )
    source_record = KSDDSourceRecord(
        source_page=arguments.source_page,
        archive_url=arguments.archive_url,
        archive_sha256=actual_hash,
        archive_bytes=actual_size,
        downloaded_at=arguments.downloaded_at,
    )
    result = prepare_ksdd_dataset(
        arguments.source_root,
        arguments.output_root,
        source_record,
        LicensePolicy.load(arguments.license_policy),
        dataset_version=arguments.dataset_version,
        created_at=arguments.created_at,
        seed=arguments.seed,
        max_hash_distance=arguments.max_hash_distance,
        minimum_formal_kpi_test_samples=arguments.minimum_formal_kpi_test_samples,
    )
    print(
        json.dumps(
            {
                "dataset_version": result.manifest.dataset_version,
                "samples": len(result.samples),
                "split_counts": result.statistics["split_counts"],
                "positive_samples": result.statistics["positive_samples"],
                "hard_negative_count": result.statistics["hard_negative_count"],
                "near_duplicate_leakage_rate": result.leakage.near_duplicate_leakage_rate,
                "evaluation_status": result.statistics["evaluation_status"],
                "formal_kpi_eligible": result.statistics["formal_kpi_eligible"],
                "probe_sample_ids": list(result.probe_sample_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
