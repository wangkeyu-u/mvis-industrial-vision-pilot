"""Paired, fairness-checked candidate versus baseline comparison."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .metrics import macro_f1
from .reporting import EvaluationReport, SampleEvaluation


class FairComparisonError(ValueError):
    """Raised when reports do not share the same evaluation population."""


@dataclass(frozen=True)
class MetricComparison:
    baseline: float
    candidate: float
    delta: float
    improvement_delta: float
    improvement_ci_lower: float | None
    improvement_ci_upper: float | None
    observation_count: int
    significance_hint: str

    def to_dict(self) -> dict[str, int | float | str | None]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ModelComparisonReport:
    schema_version: str
    fair_comparison: bool
    sample_count: int
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    minimum_reliable_samples: int
    conclusion_strength: str
    notice: str
    metrics: dict[str, MetricComparison]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fair_comparison": self.fair_comparison,
            "sample_count": self.sample_count,
            "confidence_level": self.confidence_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "minimum_reliable_samples": self.minimum_reliable_samples,
            "conclusion_strength": self.conclusion_strength,
            "notice": self.notice,
            "metrics": {name: value.to_dict() for name, value in sorted(self.metrics.items())},
        }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[math.floor(quantile * (len(ordered) - 1))]


def _paired_interval(
    count: int,
    statistic: Callable[[list[int]], float],
    *,
    metric: str,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    generator = random.Random(f"mvis-comparison:{seed}:{metric}")
    values = []
    for _ in range(resamples):
        indices = [generator.randrange(count) for _ in range(count)]
        values.append(statistic(indices))
    alpha = 1.0 - confidence_level
    return _percentile(values, alpha / 2), _percentile(values, 1.0 - alpha / 2)


def _truth_identity(sample: SampleEvaluation) -> dict[str, object]:
    return sample.truth.to_dict()


def compare_model_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    confidence_level: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260808,
    minimum_reliable_samples: int = 30,
) -> ModelComparisonReport:
    """Compare paired predictions only after strict population checks."""

    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if minimum_reliable_samples <= 0:
        raise ValueError("minimum_reliable_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if baseline.summary.get("iou_threshold") != candidate.summary.get("iou_threshold"):
        raise FairComparisonError("reports use different IoU thresholds")
    baseline_by_id = {sample.sample_id: sample for sample in baseline.samples}
    candidate_by_id = {sample.sample_id: sample for sample in candidate.samples}
    if set(baseline_by_id) != set(candidate_by_id):
        raise FairComparisonError("reports contain different sample IDs")
    sample_ids = sorted(baseline_by_id)
    for sample_id in sample_ids:
        if _truth_identity(baseline_by_id[sample_id]) != _truth_identity(candidate_by_id[sample_id]):
            raise FairComparisonError(f"ground truth differs for sample {sample_id!r}")
    baseline_samples = [baseline_by_id[sample_id] for sample_id in sample_ids]
    candidate_samples = [candidate_by_id[sample_id] for sample_id in sample_ids]
    labels = sorted({sample.truth.result for sample in baseline_samples})

    def classification(indices: list[int], samples: Sequence[SampleEvaluation]) -> float:
        return macro_f1(
            [samples[index].truth.result for index in indices],
            [samples[index].prediction.result for index in indices],
            labels=labels,
        )

    metric_inputs: dict[str, tuple[list[SampleEvaluation], list[SampleEvaluation], Callable[[list[int], Sequence[SampleEvaluation]], float], bool]] = {
        "macro_f1": (baseline_samples, candidate_samples, classification, True),
        "json_schema_validity_rate": (
            baseline_samples,
            candidate_samples,
            lambda indices, samples: sum(samples[index].json_valid for index in indices) / len(indices),
            True,
        ),
        "evidence_conclusion_consistency_rate": (
            baseline_samples,
            candidate_samples,
            lambda indices, samples: sum(samples[index].evidence_conclusion_consistent for index in indices) / len(indices),
            True,
        ),
    }
    localization_pairs = [
        (base, cand)
        for base, cand in zip(baseline_samples, candidate_samples)
        if base.localization_targets > 0
    ]
    hard_negative_pairs = [
        (base, cand)
        for base, cand in zip(baseline_samples, candidate_samples)
        if base.truth.is_hard_negative
    ]
    metric_inputs["acc_at_iou"] = (
        [item[0] for item in localization_pairs],
        [item[1] for item in localization_pairs],
        lambda indices, samples: sum(samples[index].localization_hits for index in indices)
        / sum(samples[index].localization_targets for index in indices),
        True,
    )
    metric_inputs["hard_negative_false_positive_rate"] = (
        [item[0] for item in hard_negative_pairs],
        [item[1] for item in hard_negative_pairs],
        lambda indices, samples: sum(samples[index].hard_negative_false_positive for index in indices)
        / len(indices),
        False,
    )

    comparisons = {}
    any_small = False
    for metric, (base_values, candidate_values, statistic, higher_is_better) in metric_inputs.items():
        count = len(base_values)
        any_small = any_small or count < minimum_reliable_samples

        def improvement(indices: list[int]) -> float:
            base_score = statistic(indices, base_values)
            candidate_score = statistic(indices, candidate_values)
            return candidate_score - base_score if higher_is_better else base_score - candidate_score

        lower, upper = _paired_interval(
            count,
            improvement,
            metric=metric,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
        )
        base_score = float(baseline.summary[metric])
        candidate_score = float(candidate.summary[metric])
        raw_delta = candidate_score - base_score
        improvement_delta = raw_delta if higher_is_better else -raw_delta
        if count < minimum_reliable_samples:
            hint = "insufficient_sample"
        elif lower is not None and lower > 0:
            hint = "likely_improvement"
        elif upper is not None and upper < 0:
            hint = "likely_regression"
        else:
            hint = "inconclusive"
        comparisons[metric] = MetricComparison(
            base_score,
            candidate_score,
            raw_delta,
            improvement_delta,
            lower,
            upper,
            count,
            hint,
        )
    strength = "degraded_small_sample" if any_small else "standard"
    notice = (
        "Sample support is below the configured reliability floor for at least one metric; report deltas descriptively and do not claim statistical significance."
        if any_small
        else "Paired bootstrap intervals are directional evidence, not proof against dataset bias or multiple-testing effects."
    )
    return ModelComparisonReport(
        "1.0.0",
        True,
        len(sample_ids),
        confidence_level,
        resamples,
        seed,
        minimum_reliable_samples,
        strength,
        notice,
        comparisons,
    )
