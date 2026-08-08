"""KPI-01 through KPI-05 acceptance decisions with explicit reasons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AcceptanceThresholds:
    macro_f1_min: float = 0.82
    macro_f1_improvement_min: float = 0.05
    acc_at_iou_min: float = 0.70
    acc_at_iou_improvement_min: float = 0.08
    json_validity_min: float = 0.99
    hard_negative_fpr_max: float = 0.10
    hard_negative_fpr_relative_reduction_min: float = 0.30
    consistency_min: float = 0.97

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AcceptanceReason:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class KPIAssessment:
    kpi_id: str
    metric: str
    status: str
    observed: float
    target: Mapping[str, float]
    baseline: float | None
    reasons: tuple[AcceptanceReason, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kpi_id": self.kpi_id,
            "metric": self.metric,
            "status": self.status,
            "observed": self.observed,
            "target": dict(self.target),
            "baseline": self.baseline,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


@dataclass(frozen=True)
class AcceptanceReport:
    status: str
    eligible_for_model_acceptance: bool
    assessments: tuple[KPIAssessment, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "eligible_for_model_acceptance": self.eligible_for_model_acceptance,
            "assessments": [assessment.to_dict() for assessment in self.assessments],
        }


def _metric(metrics: Mapping[str, int | float], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"metric {name!r} must be a finite number")
    return float(value)


def _baseline(baseline: Mapping[str, int | float] | None, name: str) -> float | None:
    if baseline is None or name not in baseline:
        return None
    return _metric(baseline, name)


def _status(failed: bool, missing: bool) -> str:
    if failed:
        return "failed"
    if missing:
        return "not_evaluable"
    return "passed"


def evaluate_kpi_acceptance(
    metrics: Mapping[str, int | float],
    *,
    baseline_metrics: Mapping[str, int | float] | None = None,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    fixture_only: bool = False,
    mock_only: bool = False,
    pilot_only: bool = False,
) -> AcceptanceReport:
    """Evaluate all five KPI gates; fixture, mock, and pilot results are ineligible."""

    values = {
        name: _metric(metrics, name)
        for name in (
            "macro_f1",
            "acc_at_iou",
            "json_schema_validity_rate",
            "hard_negative_false_positive_rate",
            "evidence_conclusion_consistency_rate",
        )
    }
    if fixture_only or mock_only or pilot_only:
        status = "fixture_only" if fixture_only else "mock_only" if mock_only else "pilot_only"
        if fixture_only:
            reason = AcceptanceReason(
                "FIXTURE_NOT_MODEL_SCORE",
                "contract fixtures validate evaluation code and must not be reported as model performance",
            )
        elif mock_only:
            reason = AcceptanceReason(
                "MOCK_NOT_MODEL_SCORE",
                "mock outputs validate system behavior and must not be reported as model performance",
            )
        else:
            reason = AcceptanceReason(
                "PILOT_DATASET_BELOW_SAMPLE_FLOOR",
                "the frozen test split is below its formal KPI sample floor; report pilot results only",
            )
        assessments = tuple(
            KPIAssessment(kpi_id, metric, status, values[metric], target, None, (reason,))
            for kpi_id, metric, target in (
                ("KPI-01", "macro_f1", {"minimum": thresholds.macro_f1_min}),
                ("KPI-02", "acc_at_iou", {"minimum": thresholds.acc_at_iou_min}),
                ("KPI-03", "json_schema_validity_rate", {"minimum": thresholds.json_validity_min}),
                ("KPI-04", "hard_negative_false_positive_rate", {"maximum": thresholds.hard_negative_fpr_max}),
                ("KPI-05", "evidence_conclusion_consistency_rate", {"minimum": thresholds.consistency_min}),
            )
        )
        return AcceptanceReport(status, False, assessments)

    assessments = []
    macro_baseline = _baseline(baseline_metrics, "macro_f1")
    reasons = []
    if values["macro_f1"] < thresholds.macro_f1_min:
        reasons.append(AcceptanceReason("BELOW_ABSOLUTE_TARGET", f"macro_f1 {values['macro_f1']:.6f} < {thresholds.macro_f1_min:.6f}"))
    if macro_baseline is None:
        reasons.append(AcceptanceReason("BASELINE_MISSING", "macro_f1 baseline is required"))
    elif values["macro_f1"] - macro_baseline < thresholds.macro_f1_improvement_min:
        reasons.append(AcceptanceReason("IMPROVEMENT_BELOW_TARGET", f"macro_f1 improvement {values['macro_f1'] - macro_baseline:.6f} < {thresholds.macro_f1_improvement_min:.6f}"))
    assessments.append(
        KPIAssessment(
            "KPI-01",
            "macro_f1",
            _status(any(reason.code != "BASELINE_MISSING" for reason in reasons), macro_baseline is None),
            values["macro_f1"],
            {"minimum": thresholds.macro_f1_min, "baseline_improvement_minimum": thresholds.macro_f1_improvement_min},
            macro_baseline,
            tuple(reasons),
        )
    )

    acc_baseline = _baseline(baseline_metrics, "acc_at_iou")
    reasons = []
    if values["acc_at_iou"] < thresholds.acc_at_iou_min:
        reasons.append(AcceptanceReason("BELOW_ABSOLUTE_TARGET", f"acc_at_iou {values['acc_at_iou']:.6f} < {thresholds.acc_at_iou_min:.6f}"))
    if acc_baseline is None:
        reasons.append(AcceptanceReason("BASELINE_MISSING", "acc_at_iou baseline is required"))
    elif values["acc_at_iou"] - acc_baseline < thresholds.acc_at_iou_improvement_min:
        reasons.append(AcceptanceReason("IMPROVEMENT_BELOW_TARGET", f"acc_at_iou improvement {values['acc_at_iou'] - acc_baseline:.6f} < {thresholds.acc_at_iou_improvement_min:.6f}"))
    assessments.append(
        KPIAssessment(
            "KPI-02",
            "acc_at_iou",
            _status(any(reason.code != "BASELINE_MISSING" for reason in reasons), acc_baseline is None),
            values["acc_at_iou"],
            {"minimum": thresholds.acc_at_iou_min, "baseline_improvement_minimum": thresholds.acc_at_iou_improvement_min},
            acc_baseline,
            tuple(reasons),
        )
    )

    reasons = () if values["json_schema_validity_rate"] >= thresholds.json_validity_min else (
        AcceptanceReason("BELOW_ABSOLUTE_TARGET", f"json validity {values['json_schema_validity_rate']:.6f} < {thresholds.json_validity_min:.6f}"),
    )
    assessments.append(KPIAssessment("KPI-03", "json_schema_validity_rate", "passed" if not reasons else "failed", values["json_schema_validity_rate"], {"minimum": thresholds.json_validity_min}, None, reasons))

    fpr_baseline = _baseline(baseline_metrics, "hard_negative_false_positive_rate")
    reasons_list = []
    if values["hard_negative_false_positive_rate"] > thresholds.hard_negative_fpr_max:
        reasons_list.append(AcceptanceReason("ABOVE_ABSOLUTE_TARGET", f"hard-negative FPR {values['hard_negative_false_positive_rate']:.6f} > {thresholds.hard_negative_fpr_max:.6f}"))
    if fpr_baseline is None:
        reasons_list.append(AcceptanceReason("BASELINE_MISSING", "hard-negative FPR baseline is required"))
    elif fpr_baseline <= 0:
        reasons_list.append(AcceptanceReason("BASELINE_NOT_POSITIVE", "hard-negative FPR baseline must be positive to calculate relative reduction"))
    else:
        reduction = (fpr_baseline - values["hard_negative_false_positive_rate"]) / fpr_baseline
        if reduction < thresholds.hard_negative_fpr_relative_reduction_min:
            reasons_list.append(AcceptanceReason("REDUCTION_BELOW_TARGET", f"hard-negative FPR relative reduction {reduction:.6f} < {thresholds.hard_negative_fpr_relative_reduction_min:.6f}"))
    missing_fpr = fpr_baseline is None
    assessments.append(
        KPIAssessment(
            "KPI-04",
            "hard_negative_false_positive_rate",
            _status(any(reason.code not in {"BASELINE_MISSING"} for reason in reasons_list), missing_fpr),
            values["hard_negative_false_positive_rate"],
            {"maximum": thresholds.hard_negative_fpr_max, "baseline_relative_reduction_minimum": thresholds.hard_negative_fpr_relative_reduction_min},
            fpr_baseline,
            tuple(reasons_list),
        )
    )

    reasons = () if values["evidence_conclusion_consistency_rate"] >= thresholds.consistency_min else (
        AcceptanceReason("BELOW_ABSOLUTE_TARGET", f"consistency {values['evidence_conclusion_consistency_rate']:.6f} < {thresholds.consistency_min:.6f}"),
    )
    assessments.append(KPIAssessment("KPI-05", "evidence_conclusion_consistency_rate", "passed" if not reasons else "failed", values["evidence_conclusion_consistency_rate"], {"minimum": thresholds.consistency_min}, None, reasons))

    statuses = {assessment.status for assessment in assessments}
    overall = "failed" if "failed" in statuses else "not_evaluable" if "not_evaluable" in statuses else "passed"
    return AcceptanceReport(overall, overall != "not_evaluable", tuple(assessments))
