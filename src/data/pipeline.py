"""End-to-end dataset preparation facade for algorithm and backend callers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .deduplication import DuplicateGroup, find_near_duplicate_groups, perceptual_hash
from .leakage import LeakageReport, detect_split_leakage
from .manifest import DatasetManifest, build_manifest
from .schema import DataSample, SchemaError
from .splitting import SplitRatios, entity_isolated_split
from .validation import DatasetValidationReport, validate_dataset


@dataclass(frozen=True)
class DatasetPreparationResult:
    samples: tuple[DataSample, ...]
    validation: DatasetValidationReport
    perceptual_hashes: Mapping[str, str]
    duplicate_groups: tuple[DuplicateGroup, ...]
    assignments: Mapping[str, str]
    leakage: LeakageReport
    manifest: DatasetManifest


def load_samples_jsonl(path: str | Path) -> tuple[DataSample, ...]:
    """Load canonical samples and report the exact invalid JSONL line."""

    samples = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                samples.append(DataSample.from_dict(value))
            except (json.JSONDecodeError, SchemaError, TypeError) as exc:
                raise SchemaError(f"invalid sample at JSONL line {line_number}: {exc}") from exc
    return tuple(samples)


def prepare_dataset(
    samples: Sequence[DataSample],
    dataset_root: str | Path,
    *,
    dataset_version: str,
    created_at: str,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
    max_hash_distance: int = 5,
    max_near_duplicate_leakage_rate: float = 0.005,
    schema_version: str = "1.0.0",
    frozen_test: bool = False,
    parent_version: str | None = None,
    change_summary: str | None = None,
    split_strategy_metadata: Mapping[str, Any] | None = None,
) -> DatasetPreparationResult:
    """Validate, deduplicate, split, audit, and manifest a dataset in one call."""

    canonical_samples = tuple(samples)
    validation = validate_dataset(canonical_samples, dataset_root)
    if not validation.valid:
        summary = "; ".join(
            f"{issue.sample_id or '<dataset>'}:{issue.code}" for issue in validation.errors[:10]
        )
        raise SchemaError(f"dataset validation failed: {summary}")

    root = Path(dataset_root).resolve()
    hashes = {
        sample.sample_id: perceptual_hash(root / sample.image) for sample in canonical_samples
    }
    duplicate_groups = find_near_duplicate_groups(hashes, max_distance=max_hash_distance)
    assignments = entity_isolated_split(
        canonical_samples, duplicate_groups, ratios=ratios, seed=seed
    )
    leakage = detect_split_leakage(canonical_samples, assignments, duplicate_groups)
    if not leakage.passes(max_near_duplicate_leakage_rate):
        raise SchemaError(f"split leakage gate failed: {leakage.to_dict()}")

    strategy = {
        "algorithm": "entity_and_perceptual_hash_component_greedy",
        "seed": seed,
        "ratios": ratios.as_dict(),
        "perceptual_hash": "dhash-64",
        "max_hamming_distance": max_hash_distance,
        "max_near_duplicate_leakage_rate": max_near_duplicate_leakage_rate,
    }
    strategy.update(dict(split_strategy_metadata or {}))
    manifest = build_manifest(
        canonical_samples,
        root,
        assignments,
        dataset_version=dataset_version,
        created_at=created_at,
        schema_version=schema_version,
        frozen_test=frozen_test,
        split_strategy=strategy,
        parent_version=parent_version,
        change_summary=change_summary,
    )
    return DatasetPreparationResult(
        samples=canonical_samples,
        validation=validation,
        perceptual_hashes=hashes,
        duplicate_groups=duplicate_groups,
        assignments=assignments,
        leakage=leakage,
        manifest=manifest,
    )
