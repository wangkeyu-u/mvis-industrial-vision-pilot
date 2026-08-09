"""Phase 8 step 4: entity-grouped cross-validation for the supervised U-Net.

Outer 5 folds over physical entities. Each outer fold trains on the remaining
entities' tiles only; the pixel post-processing configuration and image
threshold are selected on an entity-disjoint inner holdout (one quarter of the
outer-train entities), never on the outer test fold. Training cost bounds the
inner loop to a single holdout split; PatchCore uses the full nested 4-fold
inner loop instead. Pooled out-of-fold metrics carry bootstrap intervals and
are labeled ``internal_pilot_validation``.
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
    build_fold_assignment,
    inner_fold_assignment,
)
from src.evaluation.phase8_1_protocol import (
    FROZEN_FUSION_MODES,
    assert_entity_isolation,
    candidate_grid_fingerprint,
    frozen_postprocess_candidates,
)
from src.evaluation.phase8_cv_runner import (
    _cleanup_score_artifacts,
    _pooled_metrics,
    _records_from_scores,
)
from src.evaluation.phase8_experiments import evaluate_config
from src.inference.patchcore_specialist import select_classification_threshold
from src.training.unet_segmentation import (
    _dice_loss,
    _fuse_grid,
    _infer_images,
    _load_tile_records,
    _normalize_batch,
    _tile_arrays,
    _validation_tile_dice,
    _verify_encoder,
    build_unet,
    load_unet_config,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/models/ksdd_unet_resnet18_phase8.json"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_TILES = ROOT / "data/processed/ksdd_tiled_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8_1/entity_cv_unet"
CV_MAX_EPOCHS = 40


def _train_fold_model(
    torch: Any,
    device: Any,
    config: Any,
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    checkpoint_path: Path,
) -> dict[str, Any]:
    model = build_unet(torch, _verify_encoder(config)).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    positive_count = sum(r["label"] == "defect" for r in train_records)
    negative_count = len(train_records) - positive_count
    weights = [
        (len(train_records) / (2.0 * positive_count))
        if record["label"] == "defect"
        else (len(train_records) / (2.0 * negative_count))
        for record in train_records
    ]
    best_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(min(config.max_epochs, CV_MAX_EPOCHS)):
        sampler_generator = torch.Generator().manual_seed(config.seed + epoch)
        epoch_order = [
            int(index)
            for index in torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=len(train_records),
                replacement=True,
                generator=sampler_generator,
            )
        ]
        for batch_start in range(0, len(epoch_order), config.batch_size):
            batch_indices = epoch_order[batch_start : batch_start + config.batch_size]
            images, masks = [], []
            for index in batch_indices:
                image_array, mask_array = _tile_arrays(train_records[index], config.input_size)
                images.append(image_array)
                masks.append(mask_array)
            batch = _normalize_batch(torch, np.stack(images)).to(device)
            targets = torch.from_numpy(np.stack(masks)[:, None]).to(device)
            logits = model(batch)
            loss = config.bce_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets
            ) + config.dice_weight * _dice_loss(torch, logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation_dice = _validation_tile_dice(
            model, torch, device, validation_records, config.input_size
        )
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "best_epoch": best_epoch}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return {"model": model, "best_epoch": best_epoch, "best_validation_tile_dice": best_dice}


def _infer_fold_heatmaps(
    model: Any,
    torch: Any,
    device: Any,
    samples: Sequence[dict[str, Any]],
    config: Any,
    fusion_mode: str,
    cache_dir: Path,
    tag: str,
) -> Path:
    outputs = _infer_images(model, torch, device, samples, config)
    scores_path = cache_dir / f"{tag}_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as stream:
        for item in outputs:
            sample = item["sample"]
            fused = _fuse_grid(item, config, fusion_mode)
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


def _infer_fold_heatmaps_by_fusion(
    model: Any,
    torch: Any,
    device: Any,
    samples: Sequence[dict[str, Any]],
    config: Any,
    fusion_modes: Sequence[str],
    cache_dir: Path,
    tag: str,
) -> dict[str, Path]:
    """Run U-Net once and persist all statically allowed fusion modes."""

    outputs = _infer_images(model, torch, device, samples, config)
    paths = {mode: cache_dir / f"{tag}_{mode}_scores.jsonl" for mode in fusion_modes}
    streams = {mode: path.open("w", encoding="utf-8") for mode, path in paths.items()}
    try:
        for item in outputs:
            sample = item["sample"]
            for mode in fusion_modes:
                fused = _fuse_grid(item, config, mode)
                heatmap_path = cache_dir / f"{tag}_{mode}_{sample['sample_id']}.npy"
                np.save(heatmap_path, fused.astype(np.float32))
                streams[mode].write(
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
    finally:
        for stream in streams.values():
            stream.close()
    return paths


def run_unet_cv(
    config_path: str | Path = DEFAULT_CONFIG,
    dataset_root: str | Path = DEFAULT_DATASET,
    tiles_root: str | Path = DEFAULT_TILES,
    output_dir: str | Path = DEFAULT_OUTPUT,
    outer_folds: int = 5,
    seed: int = 20260809,
) -> dict[str, Any]:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    config = load_unet_config(config_path)
    dataset = Path(dataset_root).resolve()
    tiles_root = Path(tiles_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite U-Net CV output: {output}")
    output.mkdir(parents=True)
    cache_root = output / "fold_cache"
    cache_root.mkdir()

    fusion_modes = FROZEN_FUSION_MODES
    candidates = frozen_postprocess_candidates("unet")
    grid_fingerprint = candidate_grid_fingerprint("unet", candidates, fusion_modes)

    samples = []
    with (dataset / "samples.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            with Image.open(dataset / record["image"]) as image:
                width, height = image.size
            samples.append(
                {
                    "sample_id": record["sample_id"],
                    "entity_id": record["entity_id"],
                    "split": record["metadata"]["split"],
                    "result": record["response"]["result"],
                    "image_path": dataset / record["image"],
                    "mask_path": dataset / record["metadata"]["mask"],
                    "width": width,
                    "height": height,
                    "label_positive": record["response"]["result"] == "violation",
                }
            )
    samples.sort(key=lambda item: item["sample_id"])
    tile_records = _load_tile_records(tiles_root, config.tile_size)
    assignment = build_fold_assignment(
        samples, outer_folds=outer_folds, inner_folds=2, seed=seed
    )

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

        entity_positive = {
            entity: sum(s["label_positive"] for s in samples if s["entity_id"] == entity)
            for entity in outer_train_entities
        }
        inner_assignment = inner_fold_assignment(
            sorted(outer_train_entities),
            folds=4,
            seed=seed,
            entity_positive_counts=entity_positive,
        )
        inner_val_entities = {
            entity for entity, f in inner_assignment.items() if f == 0
        }
        inner_train_entities = outer_train_entities - inner_val_entities
        inner_train_tiles = [
            r for r in tile_records if r["entity_id"] in inner_train_entities
        ]
        inner_val_tiles = [r for r in tile_records if r["entity_id"] in inner_val_entities]
        inner_val_images = [s for s in samples if s["entity_id"] in inner_val_entities]

        assert_entity_isolation(
            outer_test_entities=outer_test_entities,
            fit_entities=inner_train_entities,
            selection_entities=inner_val_entities,
            early_stop_entities=inner_val_entities,
        )

        trained = _train_fold_model(
            torch, device, config, inner_train_tiles, inner_val_tiles,
            cache_root / f"fold{fold}_unet.pt",
        )
        model = trained["model"]

        inner_scores_paths = _infer_fold_heatmaps_by_fusion(
            model, torch, device, inner_val_images, config, fusion_modes,
            cache_root, f"fold{fold}_inner",
        )
        inner_records_by_fusion = {
            mode: _records_from_scores(path, inner_val_images)
            for mode, path in inner_scores_paths.items()
        }
        inner_records = inner_records_by_fusion[fusion_modes[0]]
        inner_scores = [record["score"] for record in inner_records]
        inner_labels = [record["label"] for record in inner_records]
        if any(inner_labels) and not all(inner_labels):
            image_threshold, _ = select_classification_threshold(inner_scores, inner_labels)
        else:
            image_threshold = max(inner_scores, default=1.0) + 1e-6

        best_config = None
        best_fusion_mode = None
        best_ranking = None
        inner_selection_metrics = []
        for fusion_mode in fusion_modes:
            for candidate in candidates:
                outcome = evaluate_config(
                    inner_records_by_fusion[fusion_mode], candidate, image_threshold
                )
                metrics = outcome["metrics"]
                inner_selection_metrics.append(
                    {
                        "fusion_mode": fusion_mode,
                        "config_fingerprint": candidate.fingerprint(),
                        "metrics": metrics,
                    }
                )
                ranking = (
                    metrics["acc_at_iou_0_5"],
                    metrics["box_f1"],
                    metrics["pixel_dice"],
                    -metrics["localization_negative_rate"],
                )
                if best_ranking is None or ranking > best_ranking:
                    best_ranking = ranking
                    best_config = candidate
                    best_fusion_mode = fusion_mode

        if best_config is None or best_fusion_mode is None:
            raise RuntimeError("inner selection produced no U-Net candidate")
        _cleanup_score_artifacts(list(inner_scores_paths.values()))
        del inner_records_by_fusion, inner_records

        test_scores_path = _infer_fold_heatmaps(
            model, torch, device, outer_test, config, best_fusion_mode,
            cache_root, f"fold{fold}_test",
        )
        test_records = _records_from_scores(test_scores_path, outer_test)
        _cleanup_score_artifacts([test_scores_path])
        outcome = evaluate_config(test_records, best_config, image_threshold)
        fold_best_epoch = trained["best_epoch"]
        fold_best_dice = trained["best_validation_tile_dice"]
        del model, trained
        if device.type == "mps":
            torch.mps.empty_cache()

        for sample_result in outcome["per_sample"]:
            sample_result["outer_fold"] = fold
            sample_result["entity_id"] = next(
                sample["entity_id"]
                for sample in outer_test
                if sample["sample_id"] == sample_result["sample_id"]
            )
        pooled_records.extend(outcome["per_sample"])
        fold_reports.append(
            {
                "outer_fold": fold,
                "test_entities": sorted(outer_test_entities),
                "test_images": len(outer_test),
                "image_threshold": image_threshold,
                "selected_fusion_mode": best_fusion_mode,
                "selected_config": asdict(best_config),
                "selected_config_fingerprint": best_config.fingerprint(),
                "best_epoch": fold_best_epoch,
                "best_inner_tile_dice": fold_best_dice,
                "metrics": outcome["metrics"],
                "selection_audit": {
                    "outer_test_entities": sorted(outer_test_entities),
                    "fit_entities": sorted(inner_train_entities),
                    "early_stop_entities": sorted(inner_val_entities),
                    "selection_entities": sorted(inner_val_entities),
                    "candidate_source": "static_protocol_grid",
                },
                "inner_selection_metrics": inner_selection_metrics,
                "seconds": round(time.perf_counter() - fold_started, 3),
            }
        )

    pooled_metrics = _pooled_metrics(pooled_records)
    report = {
        "schema_version": "1.0",
        "phase": "phase8.1",
        "kind": "entity_grouped_cv_unet",
        "protocol_label": PROTOCOL_LABEL,
        "protocol_id": assignment.protocol_id,
        "protocol_fingerprint": assignment.fingerprint(),
        "model": "unet_resnet18_supervised",
        "fusion_modes_considered": list(fusion_modes),
        "candidate_grid_fingerprint": grid_fingerprint,
        "outer_folds": outer_folds,
        "inner_loop": "single entity-grouped holdout (training-cost bound)",
        "cv_max_epochs": CV_MAX_EPOCHS,
        "seed": seed,
        "candidate_configs": [asdict(config) for config in candidates],
        "selection_scope": "inner_entities_only",
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
        "test_labels_used_for_training": False,
        "test_labels_used_for_postprocess_selection": False,
        "test_labels_used_for_route_comparison": True,
    }
    (output / "per_sample_predictions.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in pooled_records),
        encoding="utf-8",
    )
    (output / "cv_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tiles-root", type=Path, default=DEFAULT_TILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_unet_cv(
        args.config, args.dataset_root, args.tiles_root, args.output_dir
    )
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
