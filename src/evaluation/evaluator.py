"""Unified evaluation facade built from the individually tested metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .metrics import (
    BoxLike,
    acc_at_iou,
    evidence_conclusion_consistency_rate,
    hard_negative_false_positive_rate,
    json_schema_validity_rate,
    macro_f1,
)


@dataclass(frozen=True)
class EvaluationCase:
    sample_id: str
    target_label: str
    predicted_label: str
    raw_output: str | Mapping[str, Any]
    is_hard_negative: bool = False
    target_bbox: BoxLike | None = None
    predicted_bbox: BoxLike | None = None

    def __post_init__(self) -> None:
        for field_name in ("sample_id", "target_label", "predicted_label"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.is_hard_negative, bool):
            raise ValueError("is_hard_negative must be a boolean")


@dataclass(frozen=True)
class EvaluationSummary:
    sample_count: int
    localization_count: int
    hard_negative_count: int
    macro_f1: float
    acc_at_iou: float
    iou_threshold: float
    json_schema_validity_rate: float
    hard_negative_false_positive_rate: float
    evidence_conclusion_consistency_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "localization_count": self.localization_count,
            "hard_negative_count": self.hard_negative_count,
            "macro_f1": self.macro_f1,
            "acc_at_iou": self.acc_at_iou,
            "iou_threshold": self.iou_threshold,
            "json_schema_validity_rate": self.json_schema_validity_rate,
            "hard_negative_false_positive_rate": self.hard_negative_false_positive_rate,
            "evidence_conclusion_consistency_rate": self.evidence_conclusion_consistency_rate,
        }


def evaluate_cases(
    cases: Iterable[EvaluationCase],
    *,
    labels: Iterable[str] | None = None,
    iou_threshold: float = 0.5,
    positive_results: Iterable[str] = ("violation",),
    negative_results: Iterable[str] = ("compliant", "no_violation", "negative"),
    uncertain_result: str = "uncertain",
) -> EvaluationSummary:
    """Compute all milestone KPIs while preserving denominator counts."""

    records = tuple(cases)
    sample_ids = [case.sample_id for case in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("evaluation sample_id values must be unique")
    localization = [case for case in records if case.target_bbox is not None]
    payloads = [case.raw_output for case in records]
    hard_negative_mask = [case.is_hard_negative for case in records]
    positive_values = tuple(positive_results)
    negative_values = tuple(negative_results)
    return EvaluationSummary(
        sample_count=len(records),
        localization_count=len(localization),
        hard_negative_count=sum(hard_negative_mask),
        macro_f1=macro_f1(
            [case.target_label for case in records],
            [case.predicted_label for case in records],
            labels=labels,
        ),
        acc_at_iou=acc_at_iou(
            [case.predicted_bbox for case in localization],
            [case.target_bbox for case in localization],
            threshold=iou_threshold,
        ),
        iou_threshold=iou_threshold,
        json_schema_validity_rate=json_schema_validity_rate(payloads),
        hard_negative_false_positive_rate=hard_negative_false_positive_rate(
            payloads, hard_negative_mask, positive_results=positive_values
        ),
        evidence_conclusion_consistency_rate=evidence_conclusion_consistency_rate(
            payloads,
            positive_results=positive_values,
            negative_results=negative_values,
            uncertain_result=uncertain_result,
        ),
    )
