"""Reference metric implementations for the frozen evaluation set.

All aggregate functions return ``0.0`` for an empty denominator. Callers should
report the associated count so an empty slice cannot be mistaken for evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

from src.data.schema import BoundingBox

BoxLike = BoundingBox | Sequence[float]


def macro_f1(
    targets: Sequence[str], predictions: Sequence[str], labels: Iterable[str] | None = None
) -> float:
    """Unweighted mean of per-class F1, including zero-support requested labels."""

    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    selected_labels = sorted(set(labels) if labels is not None else set(targets) | set(predictions))
    if not selected_labels:
        return 0.0
    scores = []
    for label in selected_labels:
        true_positive = sum(t == label and p == label for t, p in zip(targets, predictions))
        false_positive = sum(t != label and p == label for t, p in zip(targets, predictions))
        false_negative = sum(t == label and p != label for t, p in zip(targets, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def _box_values(box: BoxLike) -> tuple[float, float, float, float]:
    if isinstance(box, BoundingBox):
        values = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
    else:
        if isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError("box must have four coordinates")
        values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("box coordinates must be finite")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("box must satisfy x1 < x2 and y1 < y2")
    return values


def intersection_over_union(prediction: BoxLike, target: BoxLike) -> float:
    """Compute IoU for two boxes using continuous ``xyxy`` geometry."""

    px1, py1, px2, py2 = _box_values(prediction)
    tx1, ty1, tx2, ty2 = _box_values(target)
    intersection_width = max(0.0, min(px2, tx2) - max(px1, tx1))
    intersection_height = max(0.0, min(py2, ty2) - max(py1, ty1))
    intersection = intersection_width * intersection_height
    prediction_area = (px2 - px1) * (py2 - py1)
    target_area = (tx2 - tx1) * (ty2 - ty1)
    union = prediction_area + target_area - intersection
    return intersection / union if union else 0.0


def acc_at_iou(
    predictions: Sequence[BoxLike | None],
    targets: Sequence[BoxLike],
    threshold: float = 0.5,
) -> float:
    """Fraction of aligned grounding cases whose IoU is at least ``threshold``.

    Each item represents one query/ground-truth target. A missing predicted box is
    a miss. Multi-object tasks must first emit one aligned case per target.
    """

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not targets:
        return 0.0
    hits = sum(
        prediction is not None and intersection_over_union(prediction, target) >= threshold
        for prediction, target in zip(predictions, targets)
    )
    return hits / len(targets)


def _parse_payload(payload: str | Mapping[str, Any]) -> Mapping[str, Any] | None:
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
    else:
        parsed = payload
    return parsed if isinstance(parsed, Mapping) else None


def validate_prediction_payload(payload: str | Mapping[str, Any]) -> bool:
    """Validate the model-owned structured result, independent of API envelope fields."""

    value = _parse_payload(payload)
    if value is None:
        return False
    required = {"result", "objects", "reason", "uncertain"}
    if not required.issubset(value):
        return False
    if not isinstance(value["result"], str) or not value["result"].strip():
        return False
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        return False
    if not isinstance(value["uncertain"], bool) or not isinstance(value["objects"], list):
        return False
    for item in value["objects"]:
        if not isinstance(item, Mapping):
            return False
        if not isinstance(item.get("label"), str) or not item["label"].strip():
            return False
        bbox = item.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(coordinate)
                for coordinate in bbox
            )
        ):
            return False
        try:
            values = _box_values(bbox)
        except ValueError:
            return False
        if values[0] < 0 or values[1] < 0:
            return False
        if "confidence" in item and (
            isinstance(item["confidence"], bool)
            or not isinstance(item["confidence"], (int, float))
            or not 0.0 <= item["confidence"] <= 1.0
        ):
            return False
        if "source" in item and (
            not isinstance(item["source"], str) or not item["source"].strip()
        ):
            return False
    return True


def json_schema_validity_rate(payloads: Sequence[str | Mapping[str, Any]]) -> float:
    """Fraction of raw outputs valid without any repair attempt."""

    return (
        sum(validate_prediction_payload(payload) for payload in payloads) / len(payloads)
        if payloads
        else 0.0
    )


def _is_false_positive(value: Mapping[str, Any] | None, positive_results: set[str]) -> bool:
    if value is None:
        return True  # Conservative: malformed hard-negative outputs cannot pass this KPI.
    result = value.get("result")
    objects = value.get("objects")
    return result in positive_results or not isinstance(objects, list) or bool(objects)


def hard_negative_false_positive_rate(
    payloads: Sequence[str | Mapping[str, Any]],
    hard_negative_mask: Sequence[bool],
    *,
    positive_results: Iterable[str] = ("violation",),
) -> float:
    """False-positive fraction over hard negatives only.

    A positive conclusion, any predicted evidence object, or malformed output counts
    as a false positive, preventing invalid JSON from artificially improving the KPI.
    """

    if len(payloads) != len(hard_negative_mask):
        raise ValueError("payloads and hard_negative_mask must have equal length")
    selected = [
        _parse_payload(payload) for payload, is_hard_negative in zip(payloads, hard_negative_mask)
        if is_hard_negative
    ]
    if not selected:
        return 0.0
    positive_set = set(positive_results)
    return sum(_is_false_positive(value, positive_set) for value in selected) / len(selected)


def _is_consistent(
    payload: str | Mapping[str, Any],
    positive_results: set[str],
    negative_results: set[str],
    uncertain_result: str,
) -> bool:
    if not validate_prediction_payload(payload):
        return False
    value = _parse_payload(payload)
    assert value is not None
    result = value["result"]
    has_evidence = bool(value["objects"])
    uncertain = value["uncertain"]
    if result in positive_results:
        return has_evidence and not uncertain
    if result in negative_results:
        return not has_evidence and not uncertain
    if result == uncertain_result:
        return uncertain and not has_evidence
    return False


def evidence_conclusion_consistency_rate(
    payloads: Sequence[str | Mapping[str, Any]],
    *,
    positive_results: Iterable[str] = ("violation",),
    negative_results: Iterable[str] = ("compliant", "no_violation", "negative"),
    uncertain_result: str = "uncertain",
) -> float:
    """Rate of schema-valid outputs whose conclusion agrees with evidence presence."""

    if not payloads:
        return 0.0
    positives, negatives = set(positive_results), set(negative_results)
    if positives & negatives or uncertain_result in positives | negatives:
        raise ValueError("positive, negative, and uncertain result labels must be disjoint")
    return sum(
        _is_consistent(payload, positives, negatives, uncertain_result) for payload in payloads
    ) / len(payloads)
