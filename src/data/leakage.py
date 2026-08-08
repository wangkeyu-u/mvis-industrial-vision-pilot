"""Post-split entity and near-duplicate leakage auditing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .deduplication import DuplicateGroup, cross_split_leakage_rate
from .schema import DataSample

VALID_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class DuplicateLeak:
    sample_ids: tuple[str, ...]
    splits: tuple[str, ...]
    max_pair_distance: int


@dataclass(frozen=True)
class LeakageReport:
    """Machine-readable evidence for the DR-004 split-isolation gate."""

    total_samples: int
    missing_sample_ids: tuple[str, ...]
    unknown_assignment_ids: tuple[str, ...]
    invalid_split_sample_ids: tuple[str, ...]
    entity_leaks: Mapping[str, tuple[str, ...]]
    duplicate_leaks: tuple[DuplicateLeak, ...]
    leaked_sample_ids: tuple[str, ...]
    near_duplicate_leakage_rate: float

    def passes(self, max_near_duplicate_rate: float = 0.005) -> bool:
        if not 0.0 <= max_near_duplicate_rate <= 1.0:
            raise ValueError("max_near_duplicate_rate must be between 0 and 1")
        return (
            not self.missing_sample_ids
            and not self.unknown_assignment_ids
            and not self.invalid_split_sample_ids
            and not self.entity_leaks
            and self.near_duplicate_leakage_rate <= max_near_duplicate_rate
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "total_samples": self.total_samples,
            "missing_sample_ids": list(self.missing_sample_ids),
            "unknown_assignment_ids": list(self.unknown_assignment_ids),
            "invalid_split_sample_ids": list(self.invalid_split_sample_ids),
            "entity_leaks": {key: list(value) for key, value in sorted(self.entity_leaks.items())},
            "duplicate_leaks": [
                {
                    "sample_ids": list(leak.sample_ids),
                    "splits": list(leak.splits),
                    "max_pair_distance": leak.max_pair_distance,
                }
                for leak in self.duplicate_leaks
            ],
            "leaked_sample_ids": list(self.leaked_sample_ids),
            "near_duplicate_leakage_rate": self.near_duplicate_leakage_rate,
        }


def detect_split_leakage(
    samples: Sequence[DataSample],
    assignments: Mapping[str, str],
    duplicate_groups: Sequence[DuplicateGroup] = (),
) -> LeakageReport:
    """Audit an arbitrary split mapping without modifying it."""

    sample_ids = {sample.sample_id for sample in samples}
    missing = tuple(sorted(sample_ids - set(assignments)))
    unknown = tuple(sorted(set(assignments) - sample_ids))
    invalid = tuple(
        sorted(
            sample_id
            for sample_id in sample_ids & set(assignments)
            if assignments[sample_id] not in VALID_SPLITS
        )
    )

    entity_splits: dict[str, set[str]] = {}
    for sample in samples:
        split = assignments.get(sample.sample_id)
        if split in VALID_SPLITS:
            entity_splits.setdefault(sample.entity_id, set()).add(split)
    entity_leaks = {
        entity_id: tuple(sorted(splits))
        for entity_id, splits in entity_splits.items()
        if len(splits) > 1
    }

    duplicate_leaks = []
    duplicate_leaked_ids: set[str] = set()
    for group in duplicate_groups:
        known_ids = tuple(sample_id for sample_id in group.sample_ids if sample_id in sample_ids)
        splits = tuple(
            sorted(
                {
                    assignments[sample_id]
                    for sample_id in known_ids
                    if assignments.get(sample_id) in VALID_SPLITS
                }
            )
        )
        if len(splits) > 1:
            duplicate_leaked_ids.update(known_ids)
            duplicate_leaks.append(DuplicateLeak(known_ids, splits, group.max_pair_distance))

    rate = cross_split_leakage_rate(duplicate_groups, assignments, len(samples))
    return LeakageReport(
        total_samples=len(samples),
        missing_sample_ids=missing,
        unknown_assignment_ids=unknown,
        invalid_split_sample_ids=invalid,
        entity_leaks=dict(sorted(entity_leaks.items())),
        duplicate_leaks=tuple(duplicate_leaks),
        leaked_sample_ids=tuple(sorted(duplicate_leaked_ids)),
        near_duplicate_leakage_rate=rate,
    )
