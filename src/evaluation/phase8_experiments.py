"""Phase 8 step 2: validation-only post-processing experiment grid.

Every experiment evaluates one PostprocessConfig on frozen heatmaps. The grid
is scored on the validation split only; the test split is evaluated exactly
once with the validation-selected winner. Each record keeps the full config,
seed, data/model fingerprints, all agreed metrics, and per-sample predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from src.evaluation.localization_postprocess import (
    PostprocessConfig,
    extract_boxes,
    greedy_box_match,
    intersection_over_union,
    scale_box_to_original,
    truth_boxes_from_mask,
    upsample_mask_nearest,
)
from src.inference.patchcore_reinfer import load_split_samples
from src.inference.performance import process_peak_memory_mb

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REINFER = ROOT / "artifacts/model/phase8/patchcore_reinfer"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8/postprocess_experiments"
DATASET_MANIFEST_SHA256 = "fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a"
EXPERIMENT_SEED = 20260809

PHASE7_BASELINE_P50_MS = 7.312
PHASE7_BASELINE_P95_MS = 15.145


def experiment_grid(
    *,
    thresholds: Sequence[tuple[str, float]] | None = None,
    max_components_values: Sequence[int] = (1, 3, 0),
    min_area_values: Sequence[int] = (1, 8, 32),
    morphologies: Sequence[str] = (
        "none",
        "open3",
        "close3",
        "close5",
        "dilate3",
        "dilate5",
        "close3_dilate3",
        "open3_close3",
    ),
    merge_options: Sequence[tuple[int, float]] = ((0, 0.0), (6, 3.0)),
) -> list[PostprocessConfig]:
    if thresholds is None:
        thresholds = [("absolute", value) for value in (0.4, 0.5, 0.6, 0.7)] + [
            ("quantile", value) for value in (0.90, 0.95, 0.97, 0.99)
        ]
    configs: list[PostprocessConfig] = []
    for mode, value in thresholds:
        for max_components in max_components_values:
            for min_area in min_area_values:
                for morphology in morphologies:
                    for merge_gap, thin_aspect in merge_options:
                        configs.append(
                            PostprocessConfig(
                                name=(
                                    f"{mode}{value}_k{max_components}_a{min_area}"
                                    f"_{morphology}_g{merge_gap}_ar{thin_aspect}"
                                ),
                                threshold_mode=mode,
                                threshold=value,
                                max_components=max_components,
                                min_area=min_area,
                                morphology=morphology,
                                merge_vertical_gap=merge_gap,
                                thin_aspect=thin_aspect,
                            )
                        )
    return configs


def load_heatmap_records(
    heatmap_source: Path,
    dataset_root: Path,
    split: str,
) -> list[dict[str, Any]]:
    """Load per-sample heatmaps, scores, truth masks at native resolution."""

    samples = {item["sample_id"]: item for item in load_split_samples(dataset_root, split)}
    index: dict[str, dict[str, Any]] = {}
    with heatmap_source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split", split) == split:
                index[record["sample_id"]] = record
    if set(index) != set(samples):
        raise ValueError(f"heatmap index and {split} samples differ")
    records = []
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        record = index[sample_id]
        heatmap = np.load(record["heatmap_npy"]) if "heatmap_npy" in record else None
        if heatmap is None:
            heatmap = np.load(record["heatmap"])
        with Image.open(sample["mask_path"]) as mask_image:
            truth_mask = (np.asarray(mask_image) > 0).astype(np.uint8)
        with Image.open(sample["image_path"]) as image:
            width, height = image.size
        records.append(
            {
                "sample_id": sample_id,
                "entity_id": sample["entity_id"],
                "label": sample["result"] == "violation",
                "score": float(record["anomaly_score"]),
                "heatmap": heatmap.astype(np.float32),
                "truth_mask": truth_mask,
                "width": width,
                "height": height,
            }
        )
    return records


def evaluate_config(
    records: Sequence[dict[str, Any]],
    config: PostprocessConfig,
    image_threshold: float,
) -> dict[str, Any]:
    pixel_tp = pixel_fp = pixel_fn = pixel_tn = 0
    box_matches = box_targets = box_predictions = box_hits = 0
    per_sample = []
    postprocess_ms = []
    negative_with_boxes = 0
    negatives = 0
    for record in records:
        started = time.perf_counter()
        boxes, binary, applied = extract_boxes(record["heatmap"], config)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        postprocess_ms.append(elapsed_ms)
        width, height = record["width"], record["height"]
        truth_mask = (np.asarray(record["truth_mask"]) > 0).astype(np.uint8)
        predicted_native = (
            upsample_mask_nearest(binary, (width, height))
            if binary.any()
            else np.zeros_like(truth_mask)
        )
        tp = int(np.logical_and(predicted_native, truth_mask).sum())
        fp = int(np.logical_and(predicted_native, 1 - truth_mask).sum())
        fn = int(np.logical_and(1 - predicted_native, truth_mask).sum())
        tn = int(np.logical_and(1 - predicted_native, 1 - truth_mask).sum())
        pixel_tp += tp
        pixel_fp += fp
        pixel_fn += fn
        pixel_tn += tn

        map_size = record["heatmap"].shape
        scaled_boxes = [
            scale_box_to_original(box, map_size, (width, height)) for box in boxes
        ]
        truth_boxes = truth_boxes_from_mask(truth_mask)
        match = greedy_box_match(truth_boxes, scaled_boxes, iou_threshold=0.5)
        box_matches += match["matches"]
        box_hits += match["target_hits"]
        box_targets += len(truth_boxes)
        box_predictions += len(scaled_boxes)
        if not record["label"]:
            negatives += 1
            if scaled_boxes:
                negative_with_boxes += 1
        per_sample.append(
            {
                "sample_id": record["sample_id"],
                "label_positive": record["label"],
                "score": record["score"],
                "classified_positive": record["score"] >= image_threshold,
                "applied_threshold": round(applied, 6),
                "predicted_boxes": [[round(v, 2) for v in box] for box in scaled_boxes],
                "truth_boxes": [[round(v, 2) for v in box] for box in truth_boxes],
                "best_iou": round(
                    max(
                        (intersection_over_union(p, t) for p in scaled_boxes for t in truth_boxes),
                        default=0.0,
                    ),
                    6,
                ),
                "pixel_tp": tp,
                "pixel_fp": fp,
                "pixel_fn": fn,
            }
        )

    scores = [record["score"] for record in records]
    labels = [record["label"] for record in records]
    image_predictions = [score >= image_threshold for score in scores]
    tp_img = sum(pred and lab for pred, lab in zip(image_predictions, labels))
    tn_img = sum((not pred) and (not lab) for pred, lab in zip(image_predictions, labels))
    fp_img = sum(pred and (not lab) for pred, lab in zip(image_predictions, labels))
    fn_img = sum((not pred) and lab for pred, lab in zip(image_predictions, labels))
    f1_pos = 2 * tp_img / (2 * tp_img + fp_img + fn_img) if 2 * tp_img + fp_img + fn_img else 0.0
    f1_neg = 2 * tn_img / (2 * tn_img + fp_img + fn_img) if 2 * tn_img + fp_img + fn_img else 0.0
    hard_negatives = sum(not label for label in labels)

    pixel_precision = pixel_tp / (pixel_tp + pixel_fp) if pixel_tp + pixel_fp else 0.0
    pixel_recall = pixel_tp / (pixel_tp + pixel_fn) if pixel_tp + pixel_fn else 0.0
    pixel_dice = (
        2 * pixel_tp / (2 * pixel_tp + pixel_fp + pixel_fn)
        if 2 * pixel_tp + pixel_fp + pixel_fn
        else 0.0
    )
    pixel_iou = (
        pixel_tp / (pixel_tp + pixel_fp + pixel_fn) if pixel_tp + pixel_fp + pixel_fn else 0.0
    )
    box_precision = box_matches / box_predictions if box_predictions else 0.0
    box_recall = box_matches / box_targets if box_targets else 0.0
    box_f1 = (
        2 * box_matches / (box_predictions + box_targets)
        if box_predictions + box_targets
        else 0.0
    )
    return {
        "metrics": {
            "image_macro_f1": (f1_pos + f1_neg) / 2.0,
            "image_auroc": _auroc(labels, scores),
            "hard_negative_fpr": fp_img / hard_negatives if hard_negatives else 0.0,
            "pixel_dice": pixel_dice,
            "pixel_iou": pixel_iou,
            "pixel_precision": pixel_precision,
            "pixel_recall": pixel_recall,
            "box_precision": box_precision,
            "box_recall": box_recall,
            "box_f1": box_f1,
            "acc_at_iou_0_5": box_hits / box_targets if box_targets else 0.0,
            "localization_negative_rate": negative_with_boxes / negatives if negatives else 0.0,
            "box_targets": box_targets,
            "box_predictions": box_predictions,
            "box_matches": box_matches,
        },
        "postprocess_p50_ms": _nearest_rank(postprocess_ms, 50),
        "postprocess_p95_ms": _nearest_rank(postprocess_ms, 95),
        "per_sample": per_sample,
    }


def _auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    concordant = 0.0
    for positive in positives:
        for negative in negatives:
            concordant += float(positive > negative) + 0.5 * float(positive == negative)
    return concordant / (len(positives) * len(negatives))


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return round(ordered[rank - 1], 3)


def select_best_config(
    experiment_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validation-only selection: Acc@IoU, then box F1, pixel Dice, lower FP box rate."""

    def ranking(record: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = record["metrics"]
        return (
            metrics["acc_at_iou_0_5"],
            metrics["box_f1"],
            metrics["pixel_dice"],
            -metrics["localization_negative_rate"],
        )

    best = max(experiment_records, key=ranking)
    return best


def run_grid(
    *,
    heatmap_source: Path,
    dataset_root: Path,
    split: str,
    image_threshold: float,
    output_path: Path,
    model_fingerprint: str,
    configs: Sequence[PostprocessConfig] | None = None,
) -> list[dict[str, Any]]:
    records = load_heatmap_records(heatmap_source, dataset_root, split)
    grid = list(configs) if configs is not None else experiment_grid()
    results = []
    for config in grid:
        outcome = evaluate_config(records, config, image_threshold)
        results.append(
            {
                "schema_version": "1.0",
                "phase": "phase8",
                "split": split,
                "config": asdict(config),
                "config_fingerprint": config.fingerprint(),
                "seed": EXPERIMENT_SEED,
                "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
                "model_fingerprint": model_fingerprint,
                "image_threshold": image_threshold,
                "metrics": outcome["metrics"],
                "postprocess_p50_ms": outcome["postprocess_p50_ms"],
                "postprocess_p95_ms": outcome["postprocess_p95_ms"],
                "estimated_p50_ms": PHASE7_BASELINE_P50_MS + outcome["postprocess_p50_ms"],
                "estimated_p95_ms": PHASE7_BASELINE_P95_MS + outcome["postprocess_p95_ms"],
                "process_peak_rss_mb": process_peak_memory_mb(),
                "per_sample": outcome["per_sample"],
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    return results


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heatmap-source", type=Path,
                        default=DEFAULT_REINFER / "scores.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-threshold", type=float, default=None)
    args = parser.parse_args(argv)

    reinfer_manifest = json.loads(
        (DEFAULT_REINFER / "run_manifest.json").read_text(encoding="utf-8")
    )
    image_threshold = (
        args.image_threshold
        if args.image_threshold is not None
        else float(reinfer_manifest["image_threshold"])
    )
    model_fingerprint = reinfer_manifest["checkpoint_sha256"]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")
    output.mkdir(parents=True)

    validation_results = run_grid(
        heatmap_source=args.heatmap_source,
        dataset_root=args.dataset_root.resolve(),
        split="validation",
        image_threshold=image_threshold,
        output_path=output / "validation_grid.jsonl",
        model_fingerprint=model_fingerprint,
    )
    best = select_best_config(validation_results)
    best_config = PostprocessConfig(**best["config"])
    test_results = run_grid(
        heatmap_source=args.heatmap_source,
        dataset_root=args.dataset_root.resolve(),
        split="test",
        image_threshold=image_threshold,
        output_path=output / "test_selected.jsonl",
        model_fingerprint=model_fingerprint,
        configs=[best_config],
    )
    selection = {
        "schema_version": "1.0",
        "phase": "phase8",
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "candidates_evaluated": len(validation_results),
        "selected_config": best["config"],
        "selected_config_fingerprint": best["config_fingerprint"],
        "validation_metrics": best["metrics"],
        "test_metrics": test_results[0]["metrics"],
        "test_postprocess_p50_ms": test_results[0]["postprocess_p50_ms"],
        "test_postprocess_p95_ms": test_results[0]["postprocess_p95_ms"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "candidates": len(validation_results),
        "selected": best["config"]["name"],
        "validation_acc_at_iou": best["metrics"]["acc_at_iou_0_5"],
        "test_acc_at_iou": test_results[0]["metrics"]["acc_at_iou_0_5"],
        "test_pixel_dice": test_results[0]["metrics"]["pixel_dice"],
        "test_box_f1": test_results[0]["metrics"]["box_f1"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
