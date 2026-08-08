"""Dataset and slice statistics shared by import reports and data cards."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .schema import DataSample


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def compute_dataset_statistics(
    samples: Sequence[DataSample], assignments: Mapping[str, str] | None = None
) -> dict[str, object]:
    results = Counter(sample.response.result for sample in samples)
    difficulties = Counter(tag for sample in samples for tag in sample.difficulty)
    sources = Counter(sample.source for sample in samples)
    licenses = Counter(sample.license.identifier for sample in samples)
    slices = Counter()
    for sample in samples:
        slices[f"result:{sample.response.result}"] += 1
        slices[f"source:{sample.source}"] += 1
        for difficulty in sample.difficulty:
            slices[f"difficulty:{difficulty}"] += 1
        metadata_slices = sample.metadata.get("slices", [])
        if isinstance(metadata_slices, list):
            for slice_name in metadata_slices:
                if isinstance(slice_name, str) and slice_name.strip():
                    slices[slice_name] += 1
    split_counts = Counter(assignments.values()) if assignments is not None else Counter()
    hard_negative_count = sum(sample.is_hard_negative for sample in samples)
    return {
        "total_samples": len(samples),
        "hard_negative_count": hard_negative_count,
        "hard_negative_rate": hard_negative_count / len(samples) if samples else 0.0,
        "result_counts": _sorted_counts(results),
        "difficulty_counts": _sorted_counts(difficulties),
        "source_counts": _sorted_counts(sources),
        "license_counts": _sorted_counts(licenses),
        "split_counts": _sorted_counts(split_counts),
        "slice_counts": _sorted_counts(slices),
    }
