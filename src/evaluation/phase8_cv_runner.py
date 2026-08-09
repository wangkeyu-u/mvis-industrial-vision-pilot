"""Phase 8 step 4: nested entity-grouped cross-validation for tiled PatchCore.

Outer 5 folds over the 48 physical entities; each outer fold re-fits the
memory bank on the remaining entities only, selects the post-processing
configuration with inner entity-grouped folds (4) and the image threshold on
pooled inner out-of-fold scores, then evaluates the outer test entities once.
Pooled out-of-fold metrics come with bootstrap intervals. Every number is
``internal_pilot_validation`` evidence, not an external holdout.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from src.evaluation.entity_cv import (
    PROTOCOL_LABEL,
    bootstrap_metric_interval,
    build_fold_assignment,
    inner_fold_assignment,
)
from src.evaluation.localization_postprocess import (
    PostprocessConfig,
)
from src.evaluation.phase8_experiments import evaluate_config
from src.inference.patchcore_specialist import select_classification_threshold
from src.inference.patchcore_tiled import (
    TiledPatchcoreConfig,
    _fuse_tile_heatmaps,
    _infer_tiles,
    _load_samples,
    _tile_crop,
    _verify_backbone,
    load_tiled_config,
    tile_offsets,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/models/ksdd_patchcore_resnet18_tiled_phase8.json"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_TILED_EXPERIMENTS = ROOT / "artifacts/model/phase8/tiled_postprocess_experiments"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8/entity_cv_patchcore"
CV_CORESET_RATIO = 0.001
CV_CANDIDATE_COUNT = 8


def load_cv_candidates(experiments_dir: Path, fusion_mode: str) -> list[PostprocessConfig]:
    """Top validation configs of the selected fusion mode plus the winner."""

    selection = json.loads((experiments_dir / "selection.json").read_text(encoding="utf-8"))
    winner = PostprocessConfig(**selection["selected_config"])
    grid_path = experiments_dir / f"validation_grid_{fusion_mode}.jsonl"
    records = []
    with grid_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))

    def ranking(record: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = record["metrics"]
        return (
            metrics["acc_at_iou_0_5"],
            metrics["box_f1"],
            metrics["pixel_dice"],
            -metrics["localization_negative_rate"],
        )

    records.sort(key=ranking, reverse=True)
    configs = []
    seen = set()
    for record in records:
        config = PostprocessConfig(**record["config"])
        if config.fingerprint() in seen:
            continue
        configs.append(config)
        seen.add(config.fingerprint())
        if len(configs) >= CV_CANDIDATE_COUNT:
            break
    if winner.fingerprint() not in seen:
        configs.append(winner)
    return configs


def _fit_memory_bank(
    module: Any,
    model: Any,
    torch: Any,
    device: Any,
    train_normal: Sequence[dict[str, Any]],
    config: TiledPatchcoreConfig,
) -> None:
    model.to(device)
    model.train()
    for sample in train_normal:
        with Image.open(sample["image_path"]) as image:
            rgb = image.convert("RGB")
            offsets = tile_offsets(rgb.size[1], config.tile_size)
            tensors = [
                _tile_crop(rgb, offset, config.tile_size, config.tile_resize)
                for offset in offsets
            ]
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            embedding = model(batch)
        model.embedding_store[-1] = embedding.detach().cpu()
        del embedding, batch, tensors
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    module.fit()


def _infer_split_heatmaps(
    model: Any,
    torch: Any,
    device: Any,
    samples: Sequence[dict[str, Any]],
    config: TiledPatchcoreConfig,
    fusion_mode: str,
    cache_dir: Path,
    tag: str,
) -> Path:
    """Infer fused heatmaps and return a flat scores.jsonl path."""

    model.to(device)
    model.eval()
    outputs = _infer_tiles(model, torch, device, samples, config)
    scores_path = cache_dir / f"{tag}_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as stream:
        for item in outputs:
            sample = item["sample"]
            fused = _fuse_tile_heatmaps(
                item["tile_maps"],
                item["offsets"],
                (item["width"], item["height"]),
                config.tile_size,
                fusion_mode,
            )
            heatmap_path = cache_dir / f"{tag}_{sample['sample_id']}.npy"
            np.save(heatmap_path, fused.astype(np.float32))
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "anomaly_score": item["score"],
                        "heatmap_npy": str(heatmap_path),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return scores_path


def _build_module(torch: Any, config: TiledPatchcoreConfig, coreset_ratio: float) -> tuple[Any, Any]:
    from anomalib.models import Patchcore

    module = Patchcore(
        backbone=config.backbone,
        layers=config.layers,
        pre_trained=True,
        coreset_sampling_ratio=coreset_ratio,
        num_neighbors=config.num_neighbors,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    return module, module.model


def run_nested_cv(
    config_path: str | Path = DEFAULT_CONFIG,
    dataset_root: str | Path = DEFAULT_DATASET,
    experiments_dir: str | Path = DEFAULT_TILED_EXPERIMENTS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 20260809,
) -> dict[str, Any]:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    config = load_tiled_config(config_path)
    dataset = Path(dataset_root).resolve()
    experiments = Path(experiments_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite CV output: {output}")
    output.mkdir(parents=True)
    cache_root = output / "fold_cache"
    cache_root.mkdir()

    selection = json.loads((experiments / "selection.json").read_text(encoding="utf-8"))
    fusion_mode = selection["selected_fusion_mode"]
    candidates = load_cv_candidates(experiments, fusion_mode)

    samples = _load_samples(dataset)
    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            sample["width"], sample["height"] = image.size
        sample["label_positive"] = sample["result"] == "violation"
    assignment = build_fold_assignment(
        samples, outer_folds=outer_folds, inner_folds=inner_folds, seed=seed
    )
    _verify_backbone(config)

    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    fold_reports = []
    pooled_records: list[dict[str, Any]] = []

    for fold in range(outer_folds):
        fold_started = time.perf_counter()
        outer_test_entities = set(assignment.fold_entities[fold])
        outer_train_entities = set(assignment.entity_fold) - outer_test_entities
        outer_test = [s for s in samples if s["entity_id"] in outer_test_entities]
        outer_train = [s for s in samples if s["entity_id"] in outer_train_entities]
        outer_train_normal = [s for s in outer_train if not s["label_positive"]]

        entity_positive = {
            entity: sum(s["label_positive"] for s in outer_train if s["entity_id"] == entity)
            for entity in outer_train_entities
        }
        inner_assignment = inner_fold_assignment(
            sorted(outer_train_entities),
            folds=inner_folds,
            seed=seed,
            entity_positive_counts=entity_positive,
        )

        inner_scores_all: list[float] = []
        inner_labels_all: list[bool] = []
        inner_pools: list[dict[str, Any]] = []
        for inner_fold in range(inner_folds):
            inner_val_entities = {
                entity for entity, f in inner_assignment.items() if f == inner_fold
            }
            inner_train_entities = outer_train_entities - inner_val_entities
            inner_train_normal = [
                s for s in outer_train_normal if s["entity_id"] in inner_train_entities
            ]
            inner_val = [s for s in outer_train if s["entity_id"] in inner_val_entities]
            module, model = _build_module(torch, config, CV_CORESET_RATIO)
            _fit_memory_bank(module, model, torch, device, inner_train_normal, config)
            scores_path = _infer_split_heatmaps(
                model, torch, device, inner_val, config, fusion_mode,
                cache_root, f"outer{fold}_inner{inner_fold}",
            )
            records = _records_from_scores(scores_path, inner_val)
            inner_pools.extend(records)
            inner_scores_all.extend(record["score"] for record in records)
            inner_labels_all.extend(record["label"] for record in records)
            del module, model
            if device.type == "mps":
                torch.mps.empty_cache()

        image_threshold, _ = select_classification_threshold(inner_scores_all, inner_labels_all)
        best_config = None
        best_ranking = None
        for candidate in candidates:
            outcome = evaluate_config(inner_pools, candidate, image_threshold)
            metrics = outcome["metrics"]
            ranking = (
                metrics["acc_at_iou_0_5"],
                metrics["box_f1"],
                metrics["pixel_dice"],
                -metrics["localization_negative_rate"],
            )
            if best_ranking is None or ranking > best_ranking:
                best_ranking = ranking
                best_config = candidate

        module, model = _build_module(torch, config, CV_CORESET_RATIO)
        _fit_memory_bank(module, model, torch, device, outer_train_normal, config)
        scores_path = _infer_split_heatmaps(
            model, torch, device, outer_test, config, fusion_mode,
            cache_root, f"outer{fold}_test",
        )
        test_records = _records_from_scores(scores_path, outer_test)
        outcome = evaluate_config(test_records, best_config, image_threshold)
        del module, model
        if device.type == "mps":
            torch.mps.empty_cache()

        for sample_result in outcome["per_sample"]:
            sample_result["outer_fold"] = fold
        pooled_records.extend(outcome["per_sample"])
        fold_reports.append(
            {
                "outer_fold": fold,
                "test_entities": sorted(outer_test_entities),
                "test_images": len(outer_test),
                "image_threshold": image_threshold,
                "selected_config": asdict(best_config),
                "selected_config_fingerprint": best_config.fingerprint(),
                "metrics": outcome["metrics"],
                "seconds": round(time.perf_counter() - fold_started, 3),
            }
        )

    pooled_metrics = _pooled_metrics(pooled_records)
    report = {
        "schema_version": "1.0",
        "phase": "phase8",
        "kind": "entity_grouped_nested_cv",
        "protocol_label": PROTOCOL_LABEL,
        "protocol_id": assignment.protocol_id,
        "protocol_fingerprint": assignment.fingerprint(),
        "model": "patchcore_tiled_resnet18",
        "fusion_mode": fusion_mode,
        "cv_coreset_ratio": CV_CORESET_RATIO,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "seed": seed,
        "candidate_configs": [asdict(config) for config in candidates],
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "fold_reports": fold_reports,
        "pooled_metrics": pooled_metrics,
        "dataset": {
            "version": config.dataset_version,
            "manifest_sha256": config.dataset_manifest_sha256,
            "images": len(samples),
            "entities": len(assignment.entity_fold),
        },
        "external_holdout": False,
        "production_acceptance": False,
    }
    (output / "per_sample_predictions.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in pooled_records),
        encoding="utf-8",
    )
    (output / "cv_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _records_from_scores(
    scores_path: Path, samples: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with scores_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                index[record["sample_id"]] = record
    records = []
    for sample in samples:
        record = index[sample["sample_id"]]
        with Image.open(sample["mask_path"]) as mask_image:
            truth_mask = (np.asarray(mask_image) > 0).astype(np.uint8)
        records.append(
            {
                "sample_id": sample["sample_id"],
                "entity_id": sample["entity_id"],
                "label": sample["label_positive"],
                "score": float(record["anomaly_score"]),
                "heatmap": np.load(record["heatmap_npy"]).astype(np.float32),
                "truth_mask": truth_mask,
                "width": sample["width"],
                "height": sample["height"],
            }
        )
    return records


def _pooled_metrics(per_sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def acc(records: Sequence[dict[str, Any]]) -> float:
        targets = [(t, p) for r in records for t in r["truth_boxes"] for p in [r["predicted_boxes"]]]
        if not targets:
            return 0.0
        hits = sum(
            any(
                _iou(pred, truth) >= 0.5 for pred in predictions
            )
            for truth, predictions in targets
        )
        return hits / len(targets)

    def image_f1(records: Sequence[dict[str, Any]]) -> float:
        tp = sum(r["label_positive"] and r["classified_positive"] for r in records)
        tn = sum((not r["label_positive"]) and (not r["classified_positive"]) for r in records)
        fp = sum((not r["label_positive"]) and r["classified_positive"] for r in records)
        fn = sum(r["label_positive"] and (not r["classified_positive"]) for r in records)
        f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
        return (f1_pos + f1_neg) / 2.0

    def pixel_dice(records: Sequence[dict[str, Any]]) -> float:
        tp = sum(r["pixel_tp"] for r in records)
        fp = sum(r["pixel_fp"] for r in records)
        fn = sum(r["pixel_fn"] for r in records)
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    def box_f1(records: Sequence[dict[str, Any]]) -> float:
        targets = sum(len(r["truth_boxes"]) for r in records)
        predictions = sum(len(r["predicted_boxes"]) for r in records)
        matches = 0
        for r in records:
            used_predictions: set[int] = set()
            used_targets: set[int] = set()
            candidates = sorted(
                (
                    (_iou(prediction, target), p_index, t_index)
                    for p_index, prediction in enumerate(r["predicted_boxes"])
                    for t_index, target in enumerate(r["truth_boxes"])
                ),
                reverse=True,
            )
            for iou, p_index, t_index in candidates:
                if iou < 0.5:
                    break
                if p_index not in used_predictions and t_index not in used_targets:
                    used_predictions.add(p_index)
                    used_targets.add(t_index)
                    matches += 1
        return 2 * matches / (predictions + targets) if predictions + targets else 0.0

    results = {}
    for name, fn in (
        ("acc_at_iou_0_5", acc),
        ("image_macro_f1", image_f1),
        ("pixel_dice", pixel_dice),
        ("box_f1", box_f1),
    ):
        interval = bootstrap_metric_interval(per_sample, fn, resamples=2000)
        results[name] = {
            "estimate": interval["mean"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "confidence": interval["confidence"],
        }
    results["n_images"] = len(per_sample)
    return results


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_TILED_EXPERIMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_nested_cv(args.config, args.dataset_root, args.experiments_dir, args.output_dir)
    print(json.dumps({
        "protocol_label": report["protocol_label"],
        "folds": report["outer_folds"],
        "pooled": {
            name: {
                "estimate": round(value["estimate"], 4),
                "ci": [round(value["ci_low"], 4), round(value["ci_high"], 4)],
            }
            for name, value in report["pooled_metrics"].items()
            if isinstance(value, dict) and "estimate" in value
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
