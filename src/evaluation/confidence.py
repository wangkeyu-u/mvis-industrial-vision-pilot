"""Deterministic bootstrap confidence intervals for the five core metrics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .metrics import macro_f1
from .reporting import EvaluationReport


@dataclass(frozen=True)
class ConfidenceInterval:
    metric: str
    point_estimate: float
    lower: float | None
    upper: float | None
    confidence_level: float
    method: str
    observation_count: int
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "metric": self.metric,
            "point_estimate": self.point_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "observation_count": self.observation_count,
            "resamples": self.resamples,
            "seed": self.seed,
        }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = math.floor(quantile * (len(ordered) - 1))
    return ordered[index]


def _bootstrap_values(
    observation_count: int,
    statistic: Callable[[list[int]], float],
    *,
    metric: str,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if observation_count == 0:
        return None, None
    generator = random.Random(f"mvis-bootstrap:{seed}:{metric}")
    results = []
    for _ in range(resamples):
        indices = [generator.randrange(observation_count) for _ in range(observation_count)]
        results.append(statistic(indices))
    alpha = 1.0 - confidence_level
    return _percentile(results, alpha / 2), _percentile(results, 1.0 - alpha / 2)


def _binary_interval(
    metric: str,
    observations: Sequence[bool],
    point_estimate: float,
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> ConfidenceInterval:
    lower, upper = _bootstrap_values(
        len(observations),
        lambda indices: sum(observations[index] for index in indices) / len(indices),
        metric=metric,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    return ConfidenceInterval(
        metric,
        point_estimate,
        lower,
        upper,
        confidence_level,
        "paired_percentile_bootstrap",
        len(observations),
        resamples,
        seed,
    )


def bootstrap_confidence_intervals(
    report: EvaluationReport,
    *,
    confidence_level: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260808,
) -> dict[str, ConfidenceInterval]:
    """Return reproducible intervals; empty metric slices have null bounds."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    samples = report.samples
    labels = sorted({sample.truth.result for sample in samples})
    macro_lower, macro_upper = _bootstrap_values(
        len(samples),
        lambda indices: macro_f1(
            [samples[index].truth.result for index in indices],
            [samples[index].prediction.result for index in indices],
            labels=labels,
        ),
        metric="macro_f1",
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    intervals = {
        "macro_f1": ConfidenceInterval(
            "macro_f1",
            float(report.summary["macro_f1"]),
            macro_lower,
            macro_upper,
            confidence_level,
            "paired_percentile_bootstrap",
            len(samples),
            resamples,
            seed,
        )
    }
    localization_observations = [
        outcome
        for sample in samples
        for outcome in (
            [True] * sample.localization_hits
            + [False] * (sample.localization_targets - sample.localization_hits)
        )
    ]
    hard_negative_observations = [
        sample.hard_negative_false_positive for sample in samples if sample.truth.is_hard_negative
    ]
    binary_metrics = {
        "acc_at_iou": localization_observations,
        "json_schema_validity_rate": [sample.json_valid for sample in samples],
        "hard_negative_false_positive_rate": hard_negative_observations,
        "evidence_conclusion_consistency_rate": [
            sample.evidence_conclusion_consistent for sample in samples
        ],
    }
    for metric, observations in binary_metrics.items():
        intervals[metric] = _binary_interval(
            metric,
            observations,
            float(report.summary[metric]),
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
        )
    return intervals
