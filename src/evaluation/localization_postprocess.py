"""Phase 8 localization post-processing operators and original-image metrics.

All operators work on a model-resolution anomaly heatmap. Pixel metrics are
computed at original image resolution after nearest-neighbor mask alignment;
box metrics use scaled boxes in original coordinates with greedy one-to-one
IoU matching, mirroring the frozen phase 7 evaluator semantics.

Parameter selection for any experiment must use validation data only; the
runner records the full configuration, seed, and data/model fingerprints so
every number is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """One frozen localization post-processing configuration."""

    name: str
    threshold_mode: str  # "absolute" | "quantile"
    threshold: float
    max_components: int = 0  # 0 keeps every surviving component
    min_area: int = 1  # component area in heatmap pixels
    morphology: str = "none"  # none|openK|closeK|dilateK|openK_closeK|closeK_dilateK
    merge_vertical_gap: int = 0  # heatmap px vertical gap for collinear thin-box merge
    thin_aspect: float = 0.0  # h/w ratio above which min_area is relaxed (0 disables)

    def __post_init__(self) -> None:
        if self.threshold_mode not in {"absolute", "quantile"}:
            raise ValueError("threshold_mode must be absolute or quantile")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.threshold_mode == "absolute" and not 0.0 < self.threshold < 1.0:
            raise ValueError("absolute threshold must be in (0, 1)")
        if self.threshold_mode == "quantile" and not 0.0 < self.threshold < 1.0:
            raise ValueError("quantile threshold must be in (0, 1)")
        if self.max_components < 0 or self.min_area <= 0:
            raise ValueError("max_components must be >= 0 and min_area > 0")
        if self.merge_vertical_gap < 0 or self.thin_aspect < 0.0:
            raise ValueError("merge gap and thin aspect must be non-negative")
        _parse_morphology(self.morphology)

    def fingerprint(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _parse_morphology(spec: str) -> tuple[tuple[str, int], ...]:
    if spec == "none":
        return ()
    ops = []
    for token in spec.split("_"):
        for name in ("open", "close", "dilate"):
            if token.startswith(name) and token[len(name):].isdigit():
                kernel = int(token[len(name):])
                if kernel < 1 or kernel > 15 or kernel % 2 == 0:
                    raise ValueError(f"unsupported morphology kernel: {token}")
                ops.append((name, kernel))
                break
        else:
            raise ValueError(f"unsupported morphology op: {token}")
    return tuple(ops)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("heatmap contains non-finite values")
    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def _apply_morphology(binary: np.ndarray, spec: str) -> np.ndarray:
    import cv2

    result = binary
    for name, kernel_size in _parse_morphology(spec):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if name == "open":
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif name == "close":
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        else:
            result = cv2.dilate(result, kernel)
    return result


def _component_boxes(binary: np.ndarray) -> list[tuple[Box, int]]:
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[Box, int]] = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        boxes.append(((float(x), float(y), float(x + w), float(y + h)), area))
    return boxes


def _merge_vertical(boxes: list[Box], gap: int) -> list[Box]:
    """Merge horizontally overlapping thin vertical boxes separated by <= gap."""

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for left_index in range(len(merged)):
            if changed:
                break
            for right_index in range(left_index + 1, len(merged)):
                a, b = merged[left_index], merged[right_index]
                x_overlap = min(a[2], b[2]) - max(a[0], b[0])
                if x_overlap <= 0:
                    continue
                vertical_gap = max(b[1] - a[3], a[1] - b[3])
                if vertical_gap <= gap:
                    merged[left_index] = (
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    )
                    merged.pop(right_index)
                    changed = True
                    break
    return merged


def extract_boxes(
    heatmap: np.ndarray, config: PostprocessConfig
) -> tuple[list[Box], np.ndarray, float]:
    """Return (boxes in heatmap coords, binary mask, applied absolute threshold)."""

    normalized = normalize_heatmap(heatmap)
    if config.threshold_mode == "quantile":
        applied = float(np.quantile(normalized, config.threshold))
    else:
        applied = config.threshold
    binary = (normalized >= applied).astype(np.uint8)
    if not binary.any():
        return [], binary, applied
    binary = _apply_morphology(binary, config.morphology)
    if not binary.any():
        return [], binary, applied
    components = _component_boxes(binary)
    survivors: list[Box] = []
    for box, area in components:
        height = box[3] - box[1]
        width = box[2] - box[0]
        aspect = height / max(width, 1.0)
        min_area = config.min_area
        if config.thin_aspect > 0.0 and aspect >= config.thin_aspect:
            min_area = max(1, config.min_area // 4)
        if area >= min_area:
            survivors.append(box)
    survivors.sort(key=lambda box: (box[3] - box[1]) * (box[2] - box[0]), reverse=True)
    if config.max_components > 0:
        survivors = survivors[: config.max_components]
    if config.merge_vertical_gap > 0:
        survivors = _merge_vertical(survivors, config.merge_vertical_gap)
    survivors.sort(key=lambda box: (box[1], box[0]))
    return survivors, binary, applied


def scale_box_to_original(box: Box, map_size: tuple[int, int], image_size: tuple[int, int]) -> Box:
    """Scale a heatmap-space box to original image pixels (exclusive xyxy)."""

    map_height, map_width = map_size
    image_width, image_height = image_size
    scale_x = image_width / map_width
    scale_y = image_height / map_height
    x1 = min(max(box[0] * scale_x, 0.0), float(image_width))
    y1 = min(max(box[1] * scale_y, 0.0), float(image_height))
    x2 = min(max(box[2] * scale_x, 0.0), float(image_width))
    y2 = min(max(box[3] * scale_y, 0.0), float(image_height))
    if x2 <= x1:
        x2 = min(float(image_width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_height), y1 + 1.0)
    return (x1, y1, x2, y2)


def intersection_over_union(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def truth_boxes_from_mask(mask: np.ndarray, min_area: int = 1) -> list[Box]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    return [box for box, area in _component_boxes(binary) if area >= min_area]


def greedy_box_match(
    targets: Sequence[Box], predictions: Sequence[Box], iou_threshold: float = 0.5
) -> dict[str, Any]:
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
    matches = 0
    for iou, p_index, t_index in candidates:
        if iou < iou_threshold:
            break
        if p_index not in used_predictions and t_index not in used_targets:
            used_predictions.add(p_index)
            used_targets.add(t_index)
            matches += 1
    target_hits = sum(
        any(intersection_over_union(prediction, target) >= iou_threshold for prediction in predictions)
        for target in targets
    )
    return {"matches": matches, "target_hits": target_hits}


def upsample_mask_nearest(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor alignment of a binary mask to (width, height)."""

    import cv2

    width, height = size
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
