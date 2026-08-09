"""Dependency-light adapters and metrics for tiled anomaly specialists."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.evaluation.metrics import intersection_over_union

Box = tuple[float, float, float, float]
NumberGrid = Sequence[Sequence[int | float | bool]]


@dataclass(frozen=True)
class SpecialistTileCase:
    """One specialist prediction aligned with one tile annotation."""

    tile_id: str
    source_sample_id: str
    anomaly_map: NumberGrid
    target_mask: NumberGrid
    image_score: float
    transform: Mapping[str, Any]


def normalize_specialist_prediction(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize common project/Anomalib specialist field aliases."""

    if not isinstance(payload, Mapping):
        raise ValueError("specialist prediction must be an object")
    score = next(
        (payload[name] for name in ("image_score", "pred_score", "anomaly_score") if name in payload),
        None,
    )
    heatmap = next(
        (payload[name] for name in ("anomaly_map", "heatmap", "pred_mask") if name in payload),
        None,
    )
    if score is None or heatmap is None:
        raise ValueError("prediction requires image_score/pred_score and anomaly_map/heatmap")
    numeric_score = float(score)
    if not math.isfinite(numeric_score):
        raise ValueError("specialist image score must be finite")
    normalized_heatmap = _grid(heatmap, "anomaly_map")
    return {
        "tile_id": str(payload.get("tile_id", "")),
        "source_sample_id": str(payload.get("source_sample_id", "")),
        "image_score": numeric_score,
        "anomaly_map": normalized_heatmap,
    }


def _grid(value: NumberGrid, name: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty 2D sequence")
    rows = []
    width: int | None = None
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
            raise ValueError(f"{name} rows must be non-empty sequences")
        converted = tuple(float(item) for item in row)
        if not all(math.isfinite(item) for item in converted):
            raise ValueError(f"{name} values must be finite")
        if width is None:
            width = len(converted)
        elif len(converted) != width:
            raise ValueError(f"{name} must be rectangular")
        rows.append(converted)
    return tuple(rows)


def threshold_heatmap(heatmap: NumberGrid, threshold: float) -> tuple[tuple[int, ...], ...]:
    """Threshold a finite rectangular anomaly heatmap into a binary mask."""

    if not math.isfinite(threshold):
        raise ValueError("heatmap threshold must be finite")
    values = _grid(heatmap, "heatmap")
    return tuple(tuple(int(value >= threshold) for value in row) for row in values)


def mask_to_bboxes(mask: NumberGrid, *, min_area: int = 1) -> tuple[Box, ...]:
    """Return 4-connected component boxes in exclusive xyxy coordinates."""

    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area <= 0:
        raise ValueError("min_area must be a positive integer")
    values = _grid(mask, "mask")
    height, width = len(values), len(values[0])
    active = [[bool(value) for value in row] for row in values]
    visited = [[False] * width for _ in range(height)]
    boxes = []
    for start_y in range(height):
        for start_x in range(width):
            if not active[start_y][start_x] or visited[start_y][start_x]:
                continue
            queue = deque([(start_x, start_y)])
            visited[start_y][start_x] = True
            xs, ys = [], []
            while queue:
                x, y = queue.popleft()
                xs.append(x)
                ys.append(y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and active[ny][nx]
                        and not visited[ny][nx]
                    ):
                        visited[ny][nx] = True
                        queue.append((nx, ny))
            if len(xs) >= min_area:
                boxes.append((float(min(xs)), float(min(ys)), float(max(xs) + 1), float(max(ys) + 1)))
    return tuple(sorted(boxes, key=lambda item: (item[1], item[0], item[3], item[2])))


def heatmap_to_bboxes(
    heatmap: NumberGrid, threshold: float, *, min_area: int = 1
) -> tuple[Box, ...]:
    return mask_to_bboxes(threshold_heatmap(heatmap, threshold), min_area=min_area)


def map_tile_bbox_to_original(
    bbox: Sequence[int | float],
    transform: Mapping[str, Any],
    *,
    coordinate_width: int | None = None,
    coordinate_height: int | None = None,
) -> Box:
    """Map a tile/heatmap box back to original pixels and clip to image bounds."""

    if isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise ValueError("bbox must have four coordinates")
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox values must be finite")
    if values[0] < 0 or values[1] < 0 or values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("bbox must be non-negative and satisfy x1<x2, y1<y2")
    required = (
        "offset_x",
        "offset_y",
        "tile_width",
        "tile_height",
        "original_width",
        "original_height",
    )
    try:
        offset_x, offset_y, tile_width, tile_height, original_width, original_height = (
            float(transform[name]) for name in required
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"transform must contain numeric fields {required}") from exc
    source_width = float(coordinate_width if coordinate_width is not None else tile_width)
    source_height = float(coordinate_height if coordinate_height is not None else tile_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("coordinate dimensions must be positive")
    scale_x, scale_y = tile_width / source_width, tile_height / source_height
    mapped = (
        offset_x + values[0] * scale_x,
        offset_y + values[1] * scale_y,
        offset_x + values[2] * scale_x,
        offset_y + values[3] * scale_y,
    )
    clipped = (
        min(max(mapped[0], 0.0), original_width),
        min(max(mapped[1], 0.0), original_height),
        min(max(mapped[2], 0.0), original_width),
        min(max(mapped[3], 0.0), original_height),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("mapped bbox is empty after clipping")
    return clipped


def binary_f1(targets: Sequence[bool | int], predictions: Sequence[bool | int]) -> float:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    true_positive = sum(bool(target) and bool(prediction) for target, prediction in zip(targets, predictions))
    false_positive = sum(not bool(target) and bool(prediction) for target, prediction in zip(targets, predictions))
    false_negative = sum(bool(target) and not bool(prediction) for target, prediction in zip(targets, predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def binary_auroc(targets: Sequence[bool | int], scores: Sequence[int | float]) -> float | None:
    """Compute tie-aware binary AUROC; return None when either class is absent."""

    if len(targets) != len(scores):
        raise ValueError("targets and scores must have equal length")
    pairs = []
    for target, score in zip(targets, scores):
        numeric = float(score)
        if not math.isfinite(numeric):
            raise ValueError("scores must be finite")
        pairs.append((numeric, bool(target)))
    positives = sum(target for _, target in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    concordant = 0.0
    positive_scores = [score for score, target in pairs if target]
    negative_scores = [score for score, target in pairs if not target]
    for positive in positive_scores:
        for negative in negative_scores:
            concordant += float(positive > negative) + 0.5 * float(positive == negative)
    return concordant / (positives * negatives)


def _confusion(
    targets: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]]
) -> tuple[int, int, int, int]:
    if len(targets) != len(predictions) or len(targets[0]) != len(predictions[0]):
        raise ValueError("target and prediction masks must have the same shape")
    tp = fp = fn = tn = 0
    for target_row, prediction_row in zip(targets, predictions):
        if len(target_row) != len(prediction_row):
            raise ValueError("target and prediction masks must have the same shape")
        for target, prediction in zip(target_row, prediction_row):
            t, p = bool(target), bool(prediction)
            tp += t and p
            fp += not t and p
            fn += t and not p
            tn += not t and not p
    return tp, fp, fn, tn


def _resize_grid(
    value: tuple[tuple[float, ...], ...], width: int, height: int
) -> tuple[tuple[float, ...], ...]:
    """Nearest-neighbor score-map alignment without adding a numeric dependency."""

    source_height, source_width = len(value), len(value[0])
    return tuple(
        tuple(
            value[min(y * source_height // height, source_height - 1)][min(x * source_width // width, source_width - 1)]
            for x in range(width)
        )
        for y in range(height)
    )


def pixel_metrics(targets: Sequence[NumberGrid], predictions: Sequence[NumberGrid]) -> dict[str, float | int]:
    """Micro-average pixel precision, recall, F1 and IoU over aligned masks."""

    if len(targets) != len(predictions):
        raise ValueError("target and prediction mask lists must have equal length")
    tp = fp = fn = tn = 0
    for target, prediction in zip(targets, predictions):
        target_grid = _grid(target, "target_mask")
        prediction_grid = _grid(prediction, "prediction_mask")
        case_tp, case_fp, case_fn, case_tn = _confusion(target_grid, prediction_grid)
        tp += case_tp
        fp += case_fp
        fn += case_fn
        tn += case_tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
    }


def box_detection_metrics(
    target_boxes: Sequence[Sequence[Sequence[int | float]]],
    predicted_boxes: Sequence[Sequence[Sequence[int | float]]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Greedy one-to-one box matching, including Acc@IoU over target boxes."""

    if len(target_boxes) != len(predicted_boxes):
        raise ValueError("target and predicted box lists must have equal length")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between zero and one")
    matches = total_targets = total_predictions = 0
    best_target_hits = 0
    for targets, predictions in zip(target_boxes, predicted_boxes):
        total_targets += len(targets)
        total_predictions += len(predictions)
        candidates = sorted(
            (
                (intersection_over_union(prediction, target), p_index, t_index)
                for p_index, prediction in enumerate(predictions)
                for t_index, target in enumerate(targets)
            ),
            reverse=True,
        )
        used_predictions: set[int] = set()
        used_targets: set[int] = set()
        for iou, p_index, t_index in candidates:
            if iou < iou_threshold:
                break
            if p_index not in used_predictions and t_index not in used_targets:
                used_predictions.add(p_index)
                used_targets.add(t_index)
                matches += 1
        best_target_hits += sum(
            any(intersection_over_union(prediction, target) >= iou_threshold for prediction in predictions)
            for target in targets
        )
    precision = matches / total_predictions if total_predictions else 0.0
    recall = matches / total_targets if total_targets else 0.0
    f1 = 2 * matches / (total_predictions + total_targets) if total_predictions + total_targets else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc_at_iou": best_target_hits / total_targets if total_targets else 0.0,
        "iou_threshold": iou_threshold,
        "matched_boxes": matches,
        "target_boxes": total_targets,
        "predicted_boxes": total_predictions,
    }


def evaluate_specialist_tiles(
    cases: Sequence[SpecialistTileCase],
    *,
    heatmap_threshold: float,
    image_threshold: float,
    min_component_area: int = 1,
    box_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate heatmaps and scores; image metrics aggregate tiles by source max score."""

    if not math.isfinite(image_threshold):
        raise ValueError("image_threshold must be finite")
    by_source: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    target_masks = []
    predicted_masks = []
    target_boxes = []
    predicted_boxes = []
    sample_results = []
    for case in cases:
        raw_heatmap = _grid(case.anomaly_map, "anomaly_map")
        truth = _grid(case.target_mask, "target_mask")
        heatmap = _resize_grid(raw_heatmap, len(truth[0]), len(truth))
        if not math.isfinite(float(case.image_score)):
            raise ValueError(f"image_score must be finite for {case.tile_id}")
        prediction = threshold_heatmap(heatmap, heatmap_threshold)
        truth_binary = tuple(tuple(int(bool(value)) for value in row) for row in truth)
        local_predictions = mask_to_bboxes(prediction, min_area=min_component_area)
        local_targets = mask_to_bboxes(truth_binary)
        height, width = len(heatmap), len(heatmap[0])
        original_predictions = tuple(
            map_tile_bbox_to_original(
                box,
                case.transform,
                coordinate_width=width,
                coordinate_height=height,
            )
            for box in local_predictions
        )
        original_targets = tuple(
            map_tile_bbox_to_original(
                box,
                case.transform,
                coordinate_width=width,
                coordinate_height=height,
            )
            for box in local_targets
        )
        label = bool(local_targets)
        by_source[case.source_sample_id].append((label, float(case.image_score)))
        target_masks.append(truth_binary)
        predicted_masks.append(prediction)
        target_boxes.append(local_targets)
        predicted_boxes.append(local_predictions)
        sample_results.append(
            {
                "tile_id": case.tile_id,
                "source_sample_id": case.source_sample_id,
                "target_positive": label,
                "image_score": float(case.image_score),
                "image_prediction": float(case.image_score) >= image_threshold,
                "predicted_boxes_tile": [list(box) for box in local_predictions],
                "predicted_boxes_original": [list(box) for box in original_predictions],
                "target_boxes_tile": [list(box) for box in local_targets],
                "target_boxes_original": [list(box) for box in original_targets],
            }
        )

    source_ids = sorted(by_source)
    image_targets = [any(label for label, _ in by_source[source]) for source in source_ids]
    image_scores = [max(score for _, score in by_source[source]) for source in source_ids]
    image_predictions = [score >= image_threshold for score in image_scores]
    return {
        "pilot_only": True,
        "model_metrics": {
            "image": {
                "f1": binary_f1(image_targets, image_predictions),
                "auroc": binary_auroc(image_targets, image_scores),
                "source_image_count": len(source_ids),
                "positive_source_image_count": sum(image_targets),
                "threshold": image_threshold,
            },
            "pixel": pixel_metrics(target_masks, predicted_masks),
            "box": box_detection_metrics(
                target_boxes, predicted_boxes, iou_threshold=box_iou_threshold
            ),
        },
        "heatmap_threshold": heatmap_threshold,
        "min_component_area": min_component_area,
        "sample_results": sample_results,
        "metric_scope": {
            "image": "source image; tile scores aggregated by maximum",
            "pixel": "micro-average; heatmaps nearest-neighbor aligned to truth tile resolution",
            "box": "tile-level 4-connected components with one-to-one IoU matching",
        },
        "warning": "pilot_only; fixture/mock values and nine-positive KSDD test evidence are not formal KPI scores",
    }
