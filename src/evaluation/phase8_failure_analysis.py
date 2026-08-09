"""Phase 8 step 1: per-sample localization failure analysis for PatchCore.

Reproduces the frozen phase 7 prediction path (validation-selected image and
bbox thresholds, largest-connected-component box) on the 56-image test split,
then classifies every defect positive into a concrete failure category with
visual overlays. No test label is used to tune anything; this is a read-only
diagnostic of the frozen model behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from src.evaluation.localization_postprocess import (
    PostprocessConfig,
    extract_boxes,
    intersection_over_union,
    normalize_heatmap,
    scale_box_to_original,
    truth_boxes_from_mask,
)
from src.inference.patchcore_reinfer import load_split_samples
from src.inference.patchcore_specialist import heatmap_to_bbox

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REINFER = ROOT / "artifacts/model/phase8/patchcore_reinfer"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8/failure_analysis"

PHASE7_BBOX_CONFIG = PostprocessConfig(
    name="phase7_frozen_largest_cc",
    threshold_mode="absolute",
    threshold=0.8,
    max_components=1,
    min_area=1,
)


def _load_scores(reinfer_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with (reinfer_dir / "scores.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["split"] == split:
                records[record["sample_id"]] = record
    return records


def _classify_positive(
    *,
    score: float,
    image_threshold: float,
    heatmap: np.ndarray,
    truth_boxes: list[tuple[float, float, float, float]],
    predicted_box: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
    bbox_threshold: float,
) -> tuple[str, list[str]]:
    """Assign one primary failure category plus contributing-factor flags."""

    flags: list[str] = []
    if score < image_threshold or predicted_box is None:
        return "classification_miss", flags

    map_height, map_width = heatmap.shape
    width, height = image_size
    scale_y = map_height / height
    thin_truth = any((box[3] - box[1]) * scale_y < 3.0 for box in truth_boxes)
    if thin_truth:
        flags.append("heatmap_resolution_loss")

    normalized = normalize_heatmap(heatmap)
    truth_mask_in_map = np.zeros((map_height, map_width), dtype=bool)
    for box in truth_boxes:
        x1 = int(box[0] / width * map_width)
        y1 = int(box[1] / height * map_height)
        x2 = max(x1 + 1, int(np.ceil(box[2] / width * map_width)))
        y2 = max(y1 + 1, int(np.ceil(box[3] / height * map_height)))
        truth_mask_in_map[y1:y2, x1:x2] = True
    truth_region_max = float(normalized[truth_mask_in_map].max()) if truth_mask_in_map.any() else 0.0
    if truth_region_max < bbox_threshold:
        flags.append("threshold_excludes_truth")

    candidate_boxes, _, _ = extract_boxes(heatmap, PostprocessConfig(
        name="diagnostic_all_components",
        threshold_mode="absolute",
        threshold=bbox_threshold,
        max_components=0,
        min_area=1,
    ))
    scaled_candidates = [
        scale_box_to_original(box, (map_height, map_width), image_size)
        for box in candidate_boxes
    ]
    pred = tuple(float(value) for value in predicted_box)
    best_iou = max((intersection_over_union(pred, box) for box in truth_boxes), default=0.0)

    non_largest_hit = any(
        max((intersection_over_union(candidate, truth) for truth in truth_boxes), default=0.0) >= 0.5
        for candidate in scaled_candidates[1:]
    )
    if non_largest_hit:
        return "wrong_component_selected", flags

    if len(truth_boxes) >= 2:
        covered = sum(
            intersection_over_union(pred, box) >= 0.1 for box in truth_boxes
        )
        if covered < len(truth_boxes):
            return "multi_defect_partial", flags

    pred_area = max(1.0, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    truth_union = _box_union(truth_boxes)
    truth_area = max(1.0, (truth_union[2] - truth_union[0]) * (truth_union[3] - truth_union[1]))
    contains_truth = (
        pred[0] <= truth_union[0] and pred[1] <= truth_union[1]
        and pred[2] >= truth_union[2] and pred[3] >= truth_union[3]
    )
    if contains_truth and pred_area / truth_area > 4.0:
        return "box_too_large", flags
    inside_truth = (
        truth_union[0] <= pred[0] and truth_union[1] <= pred[1]
        and truth_union[2] >= pred[2] and truth_union[3] >= pred[3]
    )
    if inside_truth and truth_area / pred_area > 4.0:
        return "box_too_small", flags

    pred_center = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
    truth_center = ((truth_union[0] + truth_union[2]) / 2.0, (truth_union[1] + truth_union[3]) / 2.0)
    diagonal = math.hypot(width, height)
    offset = math.dist(pred_center, truth_center) / diagonal
    if offset > 0.10:
        return "position_offset", flags
    if best_iou <= 0.0:
        return "no_overlap", flags
    return "low_overlap", flags


def _box_union(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return (
    min(box[0] for box in boxes),
    min(box[1] for box in boxes),
    max(box[2] for box in boxes),
    max(box[3] for box in boxes),
    )


def _render_overlay(
    image_path: Path,
    heatmap: np.ndarray,
    truth_boxes: list[tuple[float, float, float, float]],
    predicted_box: tuple[int, int, int, int] | None,
    output_path: Path,
) -> None:
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    width, height = base.size
    normalized = normalize_heatmap(heatmap)
    heat_image = Image.fromarray(np.uint8(np.clip(normalized * 255.0, 0, 255)), mode="L")
    heat_image = heat_image.resize((width, height), Image.Resampling.BILINEAR).convert("RGB")
    for canvas in (base, heat_image):
        draw = ImageDraw.Draw(canvas)
        for box in truth_boxes:
            draw.rectangle(list(box), outline=(0, 200, 0), width=2)
        if predicted_box is not None:
            draw.rectangle(list(predicted_box), outline=(220, 0, 0), width=2)
    combined = Image.new("RGB", (width * 2, height))
    combined.paste(base, (0, 0))
    combined.paste(heat_image, (width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)


def run_failure_analysis(
    reinfer_dir: str | Path = DEFAULT_REINFER,
    dataset_root: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    reinfer = Path(reinfer_dir).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite failure analysis: {output}")
    output.mkdir(parents=True)

    reinfer_manifest = json.loads((reinfer / "run_manifest.json").read_text(encoding="utf-8"))
    image_threshold = float(reinfer_manifest["image_threshold"])
    bbox_threshold = float(reinfer_manifest["bbox_threshold"])
    scores = _load_scores(reinfer, "test")
    samples = {sample["sample_id"]: sample for sample in load_split_samples(dataset, "test")}
    if set(scores) != set(samples):
        raise ValueError("reinfer scores and dataset test split differ")

    cases = []
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        record = scores[sample_id]
        heatmap = np.load(record["heatmap_npy"])
        with Image.open(sample["image_path"]) as image:
            width, height = image.size
        score = float(record["anomaly_score"])
        positive = score >= image_threshold
        predicted_box = (
            heatmap_to_bbox(normalize_heatmap(heatmap), bbox_threshold, width, height)
            if positive
            else None
        )
        with Image.open(sample["mask_path"]) as mask_image:
            truth_mask = np.asarray(mask_image)
        truth_boxes = truth_boxes_from_mask(truth_mask)
        label_positive = sample["result"] == "violation"
        best_iou = (
            max((intersection_over_union(predicted_box, box) for box in truth_boxes), default=0.0)
            if predicted_box is not None and truth_boxes
            else 0.0
        )
        category = None
        flags: list[str] = []
        if label_positive:
            category, flags = _classify_positive(
                score=score,
                image_threshold=image_threshold,
                heatmap=heatmap,
                truth_boxes=truth_boxes,
                predicted_box=predicted_box,
                image_size=(width, height),
                bbox_threshold=bbox_threshold,
            )
            _render_overlay(
                sample["image_path"],
                heatmap,
                truth_boxes,
                predicted_box,
                output / "overlays" / f"{sample_id}.png",
            )
        cases.append(
            {
                "sample_id": sample_id,
                "entity_id": sample["entity_id"],
                "label": sample["result"],
                "anomaly_score": score,
                "image_threshold": image_threshold,
                "classified_positive": positive,
                "truth_boxes": [list(box) for box in truth_boxes],
                "predicted_box": list(predicted_box) if predicted_box else None,
                "best_iou": best_iou,
                "localization_hit": best_iou >= 0.5,
                "failure_category": category,
                "flags": flags,
            }
        )

    positives = [case for case in cases if case["label"] == "violation"]
    negatives = [case for case in cases if case["label"] != "violation"]
    category_counts: dict[str, int] = {}
    for case in positives:
        category_counts[case["failure_category"]] = category_counts.get(case["failure_category"], 0) + 1
    summary = {
        "schema_version": "1.0",
        "phase": "phase8",
        "kind": "localization_failure_analysis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_reinfer_manifest": str(reinfer / "run_manifest.json"),
        "checkpoint_sha256": reinfer_manifest["checkpoint_sha256"],
        "image_threshold": image_threshold,
        "bbox_threshold": bbox_threshold,
        "test_labels_used_for_tuning": False,
        "test_images": len(cases),
        "positive_images": len(positives),
        "negative_images": len(negatives),
        "classification_true_positive": sum(case["classified_positive"] for case in positives),
        "classification_false_negative": sum(not case["classified_positive"] for case in positives),
        "hard_negative_false_positive": sum(case["classified_positive"] for case in negatives),
        "localization_hits": sum(case["localization_hit"] for case in positives),
        "acc_at_iou_0_5": (
            sum(case["localization_hit"] for case in positives) / len(positives) if positives else 0.0
        ),
        "failure_category_counts": category_counts,
        "flag_counts": {
            flag: sum(flag in case["flags"] for case in positives)
            for flag in sorted({flag for case in positives for flag in case["flags"]})
        },
    }
    (output / "failure_cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reinfer-dir", type=Path, default=DEFAULT_REINFER)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary = run_failure_analysis(args.reinfer_dir, args.dataset_root, args.output_dir)
    print(json.dumps({
        "positives": summary["positive_images"],
        "localization_hits": summary["localization_hits"],
        "failure_category_counts": summary["failure_category_counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
