"""Deterministic entity- and near-duplicate-isolated dataset splitting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

from .deduplication import DuplicateGroup
from .schema import DataSample


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split ratios must be non-negative and sum to 1")

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def entity_isolated_split(
    samples: Sequence[DataSample],
    duplicate_groups: Sequence[DuplicateGroup] = (),
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole connected components of entities/near-duplicates to one split.

    The greedy assignment minimizes absolute sample-count deviation from the requested
    ratio. Groups cannot be divided, so small datasets may not exactly match ratios.
    """

    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("sample_id values must be unique before splitting")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    parent = {sample.sample_id: sample.sample_id for sample in samples}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    entity_members: dict[str, list[str]] = {}
    for sample in samples:
        entity_members.setdefault(sample.entity_id, []).append(sample.sample_id)
    for members in entity_members.values():
        for member in members[1:]:
            union(members[0], member)
    for group in duplicate_groups:
        known = [sample_id for sample_id in group.sample_ids if sample_id in sample_by_id]
        for member in known[1:]:
            union(known[0], member)

    components: dict[str, list[str]] = {}
    for sample_id in sample_by_id:
        components.setdefault(find(sample_id), []).append(sample_id)

    def stable_order(members: list[str]) -> tuple[int, str]:
        identity = "\0".join(sorted(members))
        digest = hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest()
        return (-len(members), digest)

    ordered_groups = sorted(components.values(), key=stable_order)
    targets = {name: ratio * len(samples) for name, ratio in ratios.as_dict().items()}
    counts = {name: 0 for name in targets}
    assignments: dict[str, str] = {}

    for members in ordered_groups:
        size = len(members)

        def assignment_cost(split: str) -> tuple[float, float, str]:
            projected = counts.copy()
            projected[split] += size
            total_deviation = sum(abs(projected[name] - targets[name]) for name in targets)
            fill_ratio = counts[split] / targets[split] if targets[split] else float("inf")
            return total_deviation, fill_ratio, split

        selected = min(targets, key=assignment_cost)
        counts[selected] += size
        assignments.update({sample_id: selected for sample_id in members})
    return assignments


def assert_isolated(
    samples: Sequence[DataSample],
    assignments: Mapping[str, str],
    duplicate_groups: Sequence[DuplicateGroup] = (),
) -> None:
    """Raise when an entity or near-duplicate component crosses a split."""

    entity_splits: dict[str, set[str]] = {}
    for sample in samples:
        if sample.sample_id not in assignments:
            raise ValueError(f"sample {sample.sample_id!r} has no split assignment")
        entity_splits.setdefault(sample.entity_id, set()).add(assignments[sample.sample_id])
    leaking_entities = sorted(entity for entity, splits in entity_splits.items() if len(splits) > 1)
    if leaking_entities:
        raise ValueError(f"entities cross splits: {leaking_entities}")
    for group in duplicate_groups:
        splits = {assignments[item] for item in group.sample_ids if item in assignments}
        if len(splits) > 1:
            raise ValueError(f"near-duplicate group crosses splits: {group.sample_ids}")
