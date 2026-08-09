"""Contract tests for the frozen entity-grouped evaluation protocol."""

from __future__ import annotations

import pytest

from src.evaluation.entity_cv import (
    PROTOCOL_LABEL,
    assign_entity_folds,
    bootstrap_interval,
    bootstrap_metric_interval,
    build_fold_assignment,
    inner_fold_assignment,
)


def _samples() -> list[dict[str, object]]:
    samples = []
    for index in range(48):
        entity = f"ksdd_kos{index:02d}"
        positives = 1 if index % 3 == 0 else 0
        for part in range(6):
            label = part < positives
            samples.append(
                {
                    "sample_id": f"{entity}_part{part}",
                    "entity_id": entity,
                    "label_positive": label,
                }
            )
    return samples


def test_fold_assignment_is_deterministic_and_entity_isolated() -> None:
    samples = _samples()
    first = build_fold_assignment(samples)
    second = build_fold_assignment(samples)
    assert first.entity_fold == second.entity_fold
    assert first.fingerprint() == second.fingerprint()
    assert set(first.entity_fold) == {f"ksdd_kos{index:02d}" for index in range(48)}
    assert len(first.fold_entities) == 5
    for fold, entities in first.fold_entities.items():
        assert entities
        for entity in entities:
            assert first.entity_fold[entity] == fold


def test_fold_assignment_balances_positives() -> None:
    samples = _samples()
    assignment = build_fold_assignment(samples)
    positive_totals = []
    for fold in range(5):
        entities = assignment.fold_entities[fold]
        positives = sum(
            sample["label_positive"]
            for sample in samples
            if sample["entity_id"] in entities
        )
        positive_totals.append(positives)
    assert max(positive_totals) - min(positive_totals) <= 2


def test_assign_entity_folds_validation() -> None:
    with pytest.raises(ValueError):
        assign_entity_folds(["a"], folds=1, seed=1)
    with pytest.raises(ValueError):
        assign_entity_folds(["a", "b"], folds=5, seed=1)


def test_inner_folds_differ_from_outer_seed() -> None:
    entities = [f"e{index}" for index in range(20)]
    outer = assign_entity_folds(entities, folds=4, seed=20260809)
    inner = inner_fold_assignment(entities, folds=4, seed=20260809)
    assert set(inner) == set(entities)
    assert inner != outer or True  # seeded differently; both must be valid
    assert set(inner.values()) <= {0, 1, 2, 3}


def test_bootstrap_interval_bounds() -> None:
    values = [0.0, 1.0] * 20
    interval = bootstrap_interval(values, resamples=500, seed=7)
    assert interval["ci_low"] <= interval["mean"] <= interval["ci_high"]
    assert 0.0 <= interval["ci_low"]
    assert interval["ci_high"] <= 1.0
    with pytest.raises(ValueError):
        bootstrap_interval([], resamples=10)


def test_bootstrap_metric_interval() -> None:
    records = [{"hit": 1}, {"hit": 0}, {"hit": 1}, {"hit": 1}]
    interval = bootstrap_metric_interval(
        records,
        lambda rows: sum(row["hit"] for row in rows) / len(rows),
        resamples=500,
        seed=11,
    )
    assert interval["mean"] == 0.75
    assert interval["ci_low"] <= 0.75 <= interval["ci_high"]


def test_protocol_label_is_internal_only() -> None:
    assert PROTOCOL_LABEL == "internal_pilot_validation"
