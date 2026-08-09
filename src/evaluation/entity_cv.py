"""Phase 8 step 4: frozen entity-grouped evaluation protocol primitives.

The 56-image test split participated in phase 7/8 model selection, so no
result computed on it is an unbiased holdout. This module implements the
replacement protocol: deterministic Group K-Fold over the 48 physical
entities (an entity never crosses folds, and near-duplicate images share an
entity by construction), optional inner folds for parameter selection, and
bootstrap confidence intervals over pooled out-of-fold predictions.

All results produced with these primitives must be labeled
``internal_pilot_validation`` — never external holdout or production
acceptance.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

PROTOCOL_ID_V1 = "ksdd_entity_grouped_cv_v1"
PROTOCOL_ID = "ksdd_entity_grouped_nested_cv_v2"
PROTOCOL_LABEL = "internal_pilot_validation"


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    protocol_id: str
    outer_folds: int
    inner_folds: int
    seed: int
    entity_fold: dict[str, int]
    fold_entities: dict[int, tuple[str, ...]]

    def fingerprint(self) -> str:
        canonical = repr(sorted(self.entity_fold.items())).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _entity_sort_key(entity_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{entity_id}".encode("utf-8")).hexdigest()


def assign_entity_folds(
    entity_ids: Sequence[str],
    *,
    folds: int,
    seed: int,
    entity_positive_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Deterministic balanced fold assignment over physical entities.

    Entities are ordered by a seeded hash; when positive counts are provided,
    greedy balancing keeps per-fold positive totals as even as possible.
    """

    if folds <= 1:
        raise ValueError("folds must be >= 2")
    entities = sorted(set(entity_ids), key=lambda item: _entity_sort_key(item, seed))
    if len(entities) < folds:
        raise ValueError("fewer entities than folds")
    assignment: dict[str, int] = {}
    if entity_positive_counts is None:
        for index, entity in enumerate(entities):
            assignment[entity] = index % folds
        return assignment
    fold_loads = [0] * folds
    fold_sizes = [0] * folds
    ordered = sorted(
        entities,
        key=lambda item: (-entity_positive_counts.get(item, 0), _entity_sort_key(item, seed)),
    )
    for entity in ordered:
        target = min(
            range(folds),
            key=lambda fold: (fold_loads[fold], fold_sizes[fold], fold),
        )
        assignment[entity] = target
        fold_loads[target] += entity_positive_counts.get(entity, 0)
        fold_sizes[target] += 1
    return assignment


def build_fold_assignment(
    samples: Sequence[dict[str, Any]],
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 20260809,
) -> FoldAssignment:
    """Freeze the outer-fold assignment from sample records with entity/label."""

    entity_positive: dict[str, int] = {}
    for sample in samples:
        entity = sample["entity_id"]
        entity_positive[entity] = entity_positive.get(entity, 0) + int(
            bool(sample.get("label_positive", sample.get("result") == "violation"))
        )
    entity_fold = assign_entity_folds(
        sorted(entity_positive),
        folds=outer_folds,
        seed=seed,
        entity_positive_counts=entity_positive,
    )
    fold_entities = {
        fold: tuple(sorted(entity for entity, f in entity_fold.items() if f == fold))
        for fold in range(outer_folds)
    }
    return FoldAssignment(
        protocol_id=PROTOCOL_ID,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        entity_fold=entity_fold,
        fold_entities=fold_entities,
    )


def inner_fold_assignment(
    entity_ids: Sequence[str],
    *,
    folds: int,
    seed: int,
    entity_positive_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Inner-loop assignment, seeded independently from the outer assignment."""

    return assign_entity_folds(
        entity_ids,
        folds=folds,
        seed=seed + 7919,
        entity_positive_counts=entity_positive_counts,
    )


def bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 20260809,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap interval over per-image values."""

    if not values:
        raise ValueError("bootstrap requires at least one value")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(resamples - 1, math.floor(alpha * resamples)))
    high_index = max(0, min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1))
    return {
        "mean": sum(values) / n,
        "ci_low": estimates[low_index],
        "ci_high": estimates[high_index],
        "confidence": confidence,
        "resamples": float(resamples),
        "n": float(n),
    }


def bootstrap_metric_interval(
    per_sample: Sequence[dict[str, Any]],
    metric_fn: Callable[[Sequence[dict[str, Any]]], float],
    *,
    resamples: int = 2000,
    seed: int = 20260809,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap for a metric computed over pooled sample records."""

    if not per_sample:
        raise ValueError("bootstrap requires at least one sample")
    rng = random.Random(seed)
    n = len(per_sample)
    estimates = []
    for _ in range(resamples):
        sample = [per_sample[rng.randrange(n)] for _ in range(n)]
        estimates.append(metric_fn(sample))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(resamples - 1, math.floor(alpha * resamples)))
    high_index = max(0, min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1))
    return {
        "mean": metric_fn(per_sample),
        "ci_low": estimates[low_index],
        "ci_high": estimates[high_index],
        "confidence": confidence,
        "resamples": float(resamples),
        "n": float(n),
    }


def cluster_bootstrap_metric_interval(
    per_sample: Sequence[dict[str, Any]],
    metric_fn: Callable[[Sequence[dict[str, Any]]], float],
    *,
    cluster_key: str = "entity_id",
    resamples: int = 2000,
    seed: int = 20260810,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap a metric while preserving within-entity dependence.

    Each resample draws physical entities with replacement and carries every
    image belonging to the selected entity.  Image-level bootstrap is not a
    valid primary interval for KSDD because the eight views of one commutator
    are correlated.
    """

    if not per_sample:
        raise ValueError("cluster bootstrap requires at least one sample")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    clusters: dict[str, list[dict[str, Any]]] = {}
    for record in per_sample:
        value = record.get(cluster_key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing cluster key: {cluster_key}")
        clusters.setdefault(value, []).append(record)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sampled: list[dict[str, Any]] = []
        for _ in cluster_ids:
            sampled.extend(clusters[cluster_ids[rng.randrange(len(cluster_ids))]])
        estimates.append(metric_fn(sampled))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(resamples - 1, math.floor(alpha * resamples)))
    high_index = max(0, min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1))
    return {
        "mean": metric_fn(per_sample),
        "ci_low": estimates[low_index],
        "ci_high": estimates[high_index],
        "confidence": confidence,
        "resamples": float(resamples),
        "n": float(len(per_sample)),
        "clusters": float(len(cluster_ids)),
    }
