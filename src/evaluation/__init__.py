"""Deterministic offline metrics for visual compliance evaluation."""

from .acceptance import (
    AcceptanceReport,
    AcceptanceThresholds,
    KPIAssessment,
    evaluate_kpi_acceptance,
)
from .adapters import (
    INVALID_RESULT,
    NormalizedObject,
    NormalizedPrediction,
    normalize_analyze_response,
    normalize_model_result,
    normalize_model_result_payload,
)
from .comparison import (
    FairComparisonError,
    MetricComparison,
    ModelComparisonReport,
    compare_model_reports,
)
from .confidence import ConfidenceInterval, bootstrap_confidence_intervals
from .evaluator import EvaluationCase, EvaluationSummary, evaluate_cases
from .human_report import render_html_report, render_markdown_report
from .metrics import (
    acc_at_iou,
    evidence_conclusion_consistency_rate,
    hard_negative_false_positive_rate,
    intersection_over_union,
    json_schema_validity_rate,
    macro_f1,
    validate_prediction_payload,
)
from .package import (
    EvaluationPackageExport,
    build_failure_slices,
    collect_environment_info,
    export_evaluation_package,
)
from .reporting import (
    EvaluationReport,
    GroundTruthObject,
    GroundTruthRecord,
    OfflineEvaluationRecord,
    SampleEvaluation,
    evaluate_offline_records,
)
from .specialist import (
    SpecialistTileCase,
    binary_auroc,
    binary_f1,
    box_detection_metrics,
    evaluate_specialist_tiles,
    heatmap_to_bboxes,
    map_tile_bbox_to_original,
    mask_to_bboxes,
    normalize_specialist_prediction,
    pixel_metrics,
    threshold_heatmap,
)

__all__ = [
    "acc_at_iou",
    "AcceptanceReport",
    "AcceptanceThresholds",
    "bootstrap_confidence_intervals",
    "binary_auroc",
    "binary_f1",
    "build_failure_slices",
    "box_detection_metrics",
    "collect_environment_info",
    "compare_model_reports",
    "ConfidenceInterval",
    "EvaluationCase",
    "EvaluationPackageExport",
    "EvaluationReport",
    "EvaluationSummary",
    "FairComparisonError",
    "GroundTruthObject",
    "GroundTruthRecord",
    "INVALID_RESULT",
    "NormalizedObject",
    "NormalizedPrediction",
    "OfflineEvaluationRecord",
    "SampleEvaluation",
    "evidence_conclusion_consistency_rate",
    "evaluate_cases",
    "evaluate_kpi_acceptance",
    "evaluate_offline_records",
    "evaluate_specialist_tiles",
    "export_evaluation_package",
    "hard_negative_false_positive_rate",
    "heatmap_to_bboxes",
    "intersection_over_union",
    "json_schema_validity_rate",
    "KPIAssessment",
    "MetricComparison",
    "ModelComparisonReport",
    "macro_f1",
    "map_tile_bbox_to_original",
    "mask_to_bboxes",
    "normalize_specialist_prediction",
    "normalize_analyze_response",
    "normalize_model_result",
    "normalize_model_result_payload",
    "render_html_report",
    "render_markdown_report",
    "pixel_metrics",
    "SpecialistTileCase",
    "threshold_heatmap",
    "validate_prediction_payload",
]
