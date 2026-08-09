"""Synthetic end-to-end checks for the phase 8 experiment evaluation harness."""

from __future__ import annotations

import numpy as np

from src.evaluation.localization_postprocess import PostprocessConfig
from src.evaluation.phase8_experiments import evaluate_config, select_best_config


def _record(
    sample_id: str,
    *,
    label: bool,
    score: float,
    heatmap: np.ndarray,
    truth: np.ndarray,
) -> dict:
    return {
        "sample_id": sample_id,
        "entity_id": f"entity_{sample_id}",
        "label": label,
        "score": score,
        "heatmap": heatmap,
        "truth_mask": truth,
        "width": truth.shape[1],
        "height": truth.shape[0],
    }


def test_evaluate_config_perfect_localization() -> None:
    heatmap_pos = np.zeros((64, 64), dtype=np.float32)
    heatmap_pos[20:30, 20:30] = 1.0
    truth_pos = np.zeros((64, 64), dtype=np.uint8)
    truth_pos[20:30, 20:30] = 255
    heatmap_neg = np.zeros((64, 64), dtype=np.float32)
    truth_neg = np.zeros((64, 64), dtype=np.uint8)
    records = [
        _record("pos", label=True, score=10.0, heatmap=heatmap_pos, truth=truth_pos),
        _record("neg", label=False, score=1.0, heatmap=heatmap_neg, truth=truth_neg),
    ]
    config = PostprocessConfig(name="t", threshold_mode="absolute", threshold=0.5)
    outcome = evaluate_config(records, config, image_threshold=5.0)
    metrics = outcome["metrics"]
    assert metrics["acc_at_iou_0_5"] == 1.0
    assert metrics["box_f1"] == 1.0
    assert metrics["pixel_dice"] == 1.0
    assert metrics["image_macro_f1"] == 1.0
    assert metrics["hard_negative_fpr"] == 0.0
    assert len(outcome["per_sample"]) == 2


def test_evaluate_config_counts_false_positive_boxes() -> None:
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[5:8, 5:8] = 1.0
    truth = np.zeros((32, 32), dtype=np.uint8)
    records = [_record("neg", label=False, score=9.0, heatmap=heatmap, truth=truth)]
    config = PostprocessConfig(name="t", threshold_mode="absolute", threshold=0.5)
    outcome = evaluate_config(records, config, image_threshold=5.0)
    assert outcome["metrics"]["localization_negative_rate"] == 1.0
    assert outcome["metrics"]["box_predictions"] == 1
    assert outcome["metrics"]["hard_negative_fpr"] == 1.0


def test_select_best_config_prefers_acc_then_f1() -> None:
    def record(acc: float, f1: float, dice: float) -> dict:
        return {
            "config": {
                "name": f"acc{acc}_f1{f1}",
                "threshold_mode": "absolute",
                "threshold": 0.5,
                "max_components": 0,
                "min_area": 1,
                "morphology": "none",
                "merge_vertical_gap": 0,
                "thin_aspect": 0.0,
            },
            "metrics": {
                "acc_at_iou_0_5": acc,
                "box_f1": f1,
                "pixel_dice": dice,
                "localization_negative_rate": 0.0,
            },
        }

    candidates = [record(0.5, 0.1, 0.1), record(0.5, 0.3, 0.1), record(0.25, 0.9, 0.9)]
    best = select_best_config(candidates)
    assert best["config"]["name"] == "acc0.5_f10.3"
