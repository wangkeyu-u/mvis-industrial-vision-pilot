"""Batch evaluation, slices, failure taxonomy, and machine-readable reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .adapters import NormalizedObject, NormalizedPrediction
from .metrics import evidence_conclusion_consistency_rate, intersection_over_union, macro_f1


@dataclass(frozen=True)
class GroundTruthObject:
    label: str
    bbox: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundTruthObject":
        label = value.get("label")
        bbox = value.get("bbox")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("ground-truth object label must be non-empty")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("ground-truth bbox must have four coordinates")
        coordinates = tuple(float(coordinate) for coordinate in bbox)
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            raise ValueError("ground-truth bbox coordinates must be finite")
        x1, y1, x2, y2 = coordinates
        if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
            raise ValueError("ground-truth bbox must satisfy 0 <= x1 < x2 and 0 <= y1 < y2")
        return cls(label.strip(), coordinates)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "bbox": list(self.bbox)}


@dataclass(frozen=True)
class GroundTruthRecord:
    sample_id: str
    result: str
    objects: tuple[GroundTruthObject, ...]
    image_width: int
    image_height: int
    is_hard_negative: bool = False
    slices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id.strip()
            or not isinstance(self.result, str)
            or not self.result.strip()
        ):
            raise ValueError("sample_id and result must be non-empty")
        if (
            isinstance(self.image_width, bool)
            or not isinstance(self.image_width, int)
            or isinstance(self.image_height, bool)
            or not isinstance(self.image_height, int)
            or self.image_width <= 0
            or self.image_height <= 0
        ):
            raise ValueError("image dimensions must be positive")
        if not isinstance(self.is_hard_negative, bool):
            raise ValueError("is_hard_negative must be a boolean")
        if any(not isinstance(slice_name, str) or not slice_name.strip() for slice_name in self.slices):
            raise ValueError("slice tags must be non-empty strings")
        if self.is_hard_negative and self.objects:
            raise ValueError("hard-negative ground truth cannot contain objects")
        if any(
            obj.bbox[2] > self.image_width or obj.bbox[3] > self.image_height
            for obj in self.objects
        ):
            raise ValueError("ground-truth bbox lies outside the original image")
        if len(set(self.slices)) != len(self.slices):
            raise ValueError("slice tags must be unique")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundTruthRecord":
        response = value.get("response") if isinstance(value.get("response"), Mapping) else value
        raw_objects = response.get("objects", [])
        if not isinstance(raw_objects, list):
            raise ValueError("ground-truth objects must be an array")
        difficulty = value.get("difficulty", [])
        explicit_slices = value.get("slices", [])
        if not isinstance(difficulty, list) or not isinstance(explicit_slices, list):
            raise ValueError("difficulty and slices must be arrays")
        slices = list(explicit_slices)
        slices.extend(f"difficulty:{item}" for item in difficulty)
        if value.get("source"):
            slices.append(f"dataset_source:{value['source']}")
        is_hard_negative = value.get("is_hard_negative", "hard_negative" in difficulty)
        if not isinstance(is_hard_negative, bool):
            raise ValueError("is_hard_negative must be a boolean")
        return cls(
            sample_id=str(value.get("sample_id", "")),
            result=str(response.get("result", "")),
            objects=tuple(GroundTruthObject.from_dict(item) for item in raw_objects),
            image_width=value.get("image_width"),
            image_height=value.get("image_height"),
            is_hard_negative=is_hard_negative,
            slices=tuple(dict.fromkeys(str(item) for item in slices)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "result": self.result,
            "objects": [item.to_dict() for item in self.objects],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "is_hard_negative": self.is_hard_negative,
            "slices": list(self.slices),
        }


@dataclass(frozen=True)
class OfflineEvaluationRecord:
    truth: GroundTruthRecord
    prediction: NormalizedPrediction


@dataclass(frozen=True)
class SampleEvaluation:
    sample_id: str
    truth: GroundTruthRecord
    prediction: NormalizedPrediction
    classification_correct: bool
    localization_hits: int
    localization_targets: int
    json_valid: bool
    hard_negative_false_positive: bool
    evidence_conclusion_consistent: bool
    failure_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "truth": self.truth.to_dict(),
            "prediction": self.prediction.to_dict(),
            "metrics": {
                "classification_correct": self.classification_correct,
                "localization_hits": self.localization_hits,
                "localization_targets": self.localization_targets,
                "json_valid": self.json_valid,
                "hard_negative_false_positive": self.hard_negative_false_positive,
                "evidence_conclusion_consistent": self.evidence_conclusion_consistent,
            },
            "failure_codes": list(self.failure_codes),
        }


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: str
    summary: Mapping[str, int | float]
    slice_metrics: Mapping[str, Mapping[str, int | float]]
    samples: tuple[SampleEvaluation, ...]

    @property
    def failure_cases(self) -> tuple[SampleEvaluation, ...]:
        return tuple(sample for sample in self.samples if sample.failure_codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": dict(self.summary),
            "slice_metrics": {
                key: dict(value) for key, value in sorted(self.slice_metrics.items())
            },
            "samples": [sample.to_dict() for sample in self.samples],
            "failure_cases": [sample.to_dict() for sample in self.failure_cases],
        }


def _match_objects(
    targets: Sequence[GroundTruthObject],
    predictions: Sequence[NormalizedObject],
    iou_threshold: float,
) -> tuple[int, int]:
    candidates = sorted(
        (
            (intersection_over_union(prediction.bbox, target.bbox), target_index, prediction_index)
            for target_index, target in enumerate(targets)
            for prediction_index, prediction in enumerate(predictions)
            if prediction.label == target.label
        ),
        reverse=True,
    )
    used_targets: set[int] = set()
    used_predictions: set[int] = set()
    for iou, target_index, prediction_index in candidates:
        if iou < iou_threshold:
            break
        if target_index not in used_targets and prediction_index not in used_predictions:
            used_targets.add(target_index)
            used_predictions.add(prediction_index)
    return len(used_targets), len(used_predictions)


def _evaluate_sample(
    record: OfflineEvaluationRecord,
    iou_threshold: float,
    positive_results: set[str],
    negative_results: set[str],
    uncertain_result: str,
) -> SampleEvaluation:
    truth, prediction = record.truth, record.prediction
    boundary_errors = prediction.boundary_errors(truth.image_width, truth.image_height)
    bounded_predictions = tuple(
        item for index, item in enumerate(prediction.objects) if index not in boundary_errors
    )
    hits, used_predictions = _match_objects(truth.objects, bounded_predictions, iou_threshold)
    classification_correct = truth.result == prediction.result
    hard_negative_fp = truth.is_hard_negative and (
        not prediction.schema_valid
        or prediction.result in positive_results
        or bool(prediction.objects)
    )
    consistent = prediction.schema_valid and evidence_conclusion_consistency_rate(
        [prediction.core_payload()],
        positive_results=positive_results,
        negative_results=negative_results,
        uncertain_result=uncertain_result,
    ) == 1.0
    failures = []
    if not prediction.schema_valid:
        failures.append("INVALID_JSON")
    if not classification_correct:
        failures.append("CLASSIFICATION_ERROR")
    if hits < len(truth.objects):
        failures.append("LOCALIZATION_MISS")
    if len(bounded_predictions) > used_predictions:
        failures.append("LOCALIZATION_FALSE_POSITIVE")
    if boundary_errors:
        failures.append("BBOX_OUT_OF_BOUNDS")
    if hard_negative_fp:
        failures.append("HARD_NEGATIVE_FALSE_POSITIVE")
    if not consistent:
        failures.append("EVIDENCE_CONCLUSION_INCONSISTENT")
    if prediction.result == uncertain_result and prediction.refusal_code is None:
        failures.append("REFUSAL_MISSING")
    if prediction.result != uncertain_result and prediction.refusal_code is not None:
        failures.append("REFUSAL_UNEXPECTED")
    return SampleEvaluation(
        sample_id=truth.sample_id,
        truth=truth,
        prediction=prediction,
        classification_correct=classification_correct,
        localization_hits=hits,
        localization_targets=len(truth.objects),
        json_valid=prediction.schema_valid,
        hard_negative_false_positive=hard_negative_fp,
        evidence_conclusion_consistent=consistent,
        failure_codes=tuple(failures),
    )


def _aggregate(samples: Sequence[SampleEvaluation]) -> dict[str, int | float]:
    sample_count = len(samples)
    labels = sorted({sample.truth.result for sample in samples})
    localization_targets = sum(sample.localization_targets for sample in samples)
    localization_hits = sum(sample.localization_hits for sample in samples)
    hard_negatives = sum(sample.truth.is_hard_negative for sample in samples)
    return {
        "sample_count": sample_count,
        "localization_target_count": localization_targets,
        "hard_negative_count": hard_negatives,
        "failure_case_count": sum(bool(sample.failure_codes) for sample in samples),
        "macro_f1": macro_f1(
            [sample.truth.result for sample in samples],
            [sample.prediction.result for sample in samples],
            labels=labels,
        ),
        "acc_at_iou": localization_hits / localization_targets if localization_targets else 0.0,
        "json_schema_validity_rate": (
            sum(sample.json_valid for sample in samples) / sample_count if sample_count else 0.0
        ),
        "hard_negative_false_positive_rate": (
            sum(sample.hard_negative_false_positive for sample in samples) / hard_negatives
            if hard_negatives
            else 0.0
        ),
        "evidence_conclusion_consistency_rate": (
            sum(sample.evidence_conclusion_consistent for sample in samples) / sample_count
            if sample_count
            else 0.0
        ),
    }


def evaluate_offline_records(
    records: Iterable[OfflineEvaluationRecord],
    *,
    iou_threshold: float = 0.5,
    positive_results: Iterable[str] = ("violation",),
    negative_results: Iterable[str] = ("compliant", "no_violation", "negative"),
    uncertain_result: str = "uncertain",
) -> EvaluationReport:
    """Evaluate normalized outputs and return samples, slices, and failures."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    input_records = tuple(records)
    sample_ids = [record.truth.sample_id for record in input_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("offline evaluation sample_id values must be unique")
    positives, negatives = set(positive_results), set(negative_results)
    samples = tuple(
        _evaluate_sample(record, iou_threshold, positives, negatives, uncertain_result)
        for record in input_records
    )

    def sample_slices(sample: SampleEvaluation) -> set[str]:
        return set(sample.truth.slices) | {
            f"prediction_source:{sample.prediction.source_kind}"
        }

    slice_names = sorted({slice_name for sample in samples for slice_name in sample_slices(sample)})
    slice_metrics = {
        slice_name: _aggregate(
            [sample for sample in samples if slice_name in sample_slices(sample)]
        )
        for slice_name in slice_names
    }
    summary = _aggregate(samples)
    summary["iou_threshold"] = iou_threshold
    return EvaluationReport("1.0.0", summary, slice_metrics, samples)
