"""Contract tests for phase 8 localization post-processing primitives."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.localization_postprocess import (
    PostprocessConfig,
    extract_boxes,
    greedy_box_match,
    intersection_over_union,
    normalize_heatmap,
    scale_box_to_original,
    truth_boxes_from_mask,
    upsample_mask_nearest,
)


def _config(**overrides) -> PostprocessConfig:
    base = {
        "name": "test",
        "threshold_mode": "absolute",
        "threshold": 0.5,
    }
    base.update(overrides)
    return PostprocessConfig(**base)


def test_config_validation_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        _config(threshold=1.5)
    with pytest.raises(ValueError):
        _config(threshold_mode="quantile", threshold=0.0)
    with pytest.raises(ValueError):
        _config(threshold_mode="unsupported")
    with pytest.raises(ValueError):
        _config(min_area=0)
    with pytest.raises(ValueError):
        _config(morphology="explode3")


def test_extract_boxes_single_component() -> None:
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[10:14, 8:12] = 1.0
    boxes, binary, applied = extract_boxes(heatmap, _config())
    assert applied == 0.5
    assert binary.sum() == 16
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert (x2 - x1, y2 - y1) == (4, 4)


def test_extract_boxes_max_components_keeps_largest() -> None:
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[2:4, 2:4] = 0.9  # 4 px
    heatmap[10:20, 10:20] = 0.8  # 100 px
    boxes, _, _ = extract_boxes(heatmap, _config(max_components=1))
    assert len(boxes) == 1
    assert (boxes[0][2] - boxes[0][0]) == 10
    boxes_all, _, _ = extract_boxes(heatmap, _config(max_components=0))
    assert len(boxes_all) == 2


def test_extract_boxes_min_area_and_thin_relaxation() -> None:
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[5:40, 10:12] = 1.0  # 70 px, thin vertical (h/w = 17.5)
    strict, _, _ = extract_boxes(heatmap, _config(min_area=100))
    assert strict == []
    relaxed, _, _ = extract_boxes(heatmap, _config(min_area=100, thin_aspect=3.0))
    assert len(relaxed) == 1


def test_quantile_threshold() -> None:
    rng = np.random.default_rng(0)
    heatmap = rng.random((32, 32), dtype=np.float32)
    heatmap[0, 0] = 1.0
    boxes, _, applied = extract_boxes(
        heatmap, _config(threshold_mode="quantile", threshold=0.99)
    )
    assert applied > 0.9
    assert boxes


def test_morphology_close_merges_nearby_components() -> None:
    heatmap = np.zeros((32, 64), dtype=np.float32)
    heatmap[8:16, 10:13] = 1.0
    heatmap[8:16, 14:17] = 1.0
    plain, _, _ = extract_boxes(heatmap, _config())
    assert len(plain) == 2
    closed, _, _ = extract_boxes(heatmap, _config(morphology="close3"))
    assert len(closed) == 1


def test_merge_vertical_gap() -> None:
    heatmap = np.zeros((64, 32), dtype=np.float32)
    heatmap[5:10, 10:12] = 1.0
    heatmap[14:20, 10:12] = 1.0
    merged, _, _ = extract_boxes(heatmap, _config(merge_vertical_gap=6))
    assert len(merged) == 1
    assert merged[0][3] == 20


def test_scale_box_to_original_clips_and_scales() -> None:
    box = scale_box_to_original((10, 10, 20, 30), (64, 64), (128, 256))
    assert box == (20.0, 40.0, 40.0, 120.0)
    clipped = scale_box_to_original((60, 60, 64, 64), (64, 64), (100, 100))
    assert clipped[2] <= 100 and clipped[3] <= 100


def test_iou_and_greedy_matching() -> None:
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    match = greedy_box_match(
        [(0, 0, 10, 10), (100, 100, 110, 110)],
        [(0, 0, 10, 10), (100, 100, 108, 108)],
    )
    assert match["matches"] == 2
    assert match["target_hits"] == 2


def test_truth_boxes_and_upsample() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 3:9] = 255
    boxes = truth_boxes_from_mask(mask)
    assert boxes == [(3.0, 2.0, 9.0, 5.0)]
    up = upsample_mask_nearest((mask > 0).astype(np.uint8), (40, 40))
    assert up.shape == (40, 40)
    assert int(up.sum()) == 18 * 4


def test_normalize_heatmap_constant() -> None:
    flat = np.ones((8, 8), dtype=np.float32)
    normalized = normalize_heatmap(flat)
    assert float(normalized.max()) == 0.0
