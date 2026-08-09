"""Phase 8: tiled PatchCore with lossless tile-to-original coordinate mapping.

Phase 7 resized the full ~500x1265 image to 256x256, a ~5x vertical resolution
loss that smears thin scratch defects into single blobs. This module fits
PatchCore on deterministic square tiles (full width, top/middle/bottom with
overlap), upsamples each tile anomaly map back to tile pixels, and fuses the
overlapping tile maps at native image resolution with max / mean / weighted
(blend by distance to tile center) rules. Tile offsets are pure translations,
so heatmap coordinates map back to original pixels without any resampling of
the truth annotation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .patchcore_specialist import (
    _mps_driver_memory,
    _sha256,
    normalize_heatmap,
    select_classification_threshold,
)
from .performance import process_peak_memory_mb
from .real_probe import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/models/ksdd_patchcore_resnet18_tiled_phase8.json"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8/patchcore_tiled"


@dataclass(frozen=True, slots=True)
class TiledPatchcoreConfig:
    schema_version: str
    experiment_id: str
    implementation: str
    implementation_version: str
    algorithm: str
    backbone: str
    backbone_repository: str
    backbone_revision: str
    backbone_sha256: str
    backbone_bytes: int
    layers: tuple[str, ...]
    tile_size: int
    tile_resize: int
    coreset_sampling_ratio: float
    num_neighbors: int
    seed: int
    device: str
    dataset_version: str
    dataset_manifest_sha256: str
    expected_train_normal: int
    expected_validation: int
    expected_test: int
    fusion_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported tiled PatchCore config schema_version")
        if self.implementation != "anomalib" or self.algorithm != "patchcore":
            raise ValueError("phase 8 tiled specialist must use official Anomalib PatchCore")
        if len(self.backbone_sha256) != 64 or len(self.dataset_manifest_sha256) != 64:
            raise ValueError("configured hashes must be SHA-256")
        if self.tile_size <= 0 or self.tile_resize <= 0:
            raise ValueError("tile sizes must be positive")
        if not 0.0 < self.coreset_sampling_ratio <= 1.0:
            raise ValueError("coreset_sampling_ratio must be in (0, 1]")
        unknown = set(self.fusion_modes) - {"max", "mean", "weighted"}
        if unknown or not self.fusion_modes:
            raise ValueError(f"unsupported fusion modes: {sorted(unknown)}")


def load_tiled_config(path: str | Path) -> TiledPatchcoreConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = set(TiledPatchcoreConfig.__dataclass_fields__)
    if set(payload) != required:
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        raise ValueError(f"tiled config fields differ; missing={missing}, unknown={unknown}")
    payload["layers"] = tuple(payload["layers"])
    payload["fusion_modes"] = tuple(payload["fusion_modes"])
    return TiledPatchcoreConfig(**payload)


def tile_offsets(image_height: int, tile_size: int) -> tuple[int, ...]:
    """Deterministic top/middle/bottom vertical offsets with maximal overlap."""

    if image_height <= tile_size:
        return (0,)
    middle = (image_height - tile_size) // 2
    bottom = image_height - tile_size
    return tuple(sorted({0, middle, bottom}))


def _tile_crop(image: Image.Image, offset_y: int, tile_size: int, resize: int) -> Any:
    import torch

    crop = image.crop((0, offset_y, image.width, offset_y + tile_size))
    resized = crop.resize((resize, resize), Image.Resampling.BILINEAR)
    values = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def _fuse_tile_heatmaps(
    tile_maps: Sequence[np.ndarray],
    offsets: Sequence[int],
    image_size: tuple[int, int],
    tile_size: int,
    mode: str,
) -> np.ndarray:
    """Upsample tile maps to tile pixels and fuse at native resolution."""

    import cv2

    width, height = image_size
    accumulator = np.zeros((height, width), dtype=np.float32)
    if mode == "max":
        for tile_map, offset in zip(tile_maps, offsets):
            upsampled = cv2.resize(tile_map, (width, tile_size), interpolation=cv2.INTER_LINEAR)
            accumulator[offset : offset + tile_size] = np.maximum(
                accumulator[offset : offset + tile_size], upsampled
            )
        return accumulator
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for tile_map, offset in zip(tile_maps, offsets):
        upsampled = cv2.resize(tile_map, (width, tile_size), interpolation=cv2.INTER_LINEAR)
        if mode == "mean":
            weights = np.ones((tile_size, width), dtype=np.float32)
        elif mode == "weighted":
            center = tile_size / 2.0
            vertical = 1.0 - np.abs(np.arange(tile_size) - center) / center
            weights = np.repeat(vertical[:, None] + 1e-3, width, axis=1).astype(np.float32)
        else:
            raise ValueError(f"unsupported fusion mode: {mode}")
        accumulator[offset : offset + tile_size] += upsampled * weights
        weight_sum[offset : offset + tile_size] += weights
    return accumulator / np.maximum(weight_sum, 1e-6)


def _verify_backbone(config: TiledPatchcoreConfig) -> Path:
    root = Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshot = (
        root / "models--timm--resnet18.a1_in1k" / "snapshots" / config.backbone_revision
    )
    for path in (item for item in snapshot.rglob("*") if item.is_file()):
        if path.stat().st_size == config.backbone_bytes and _sha256(path) == config.backbone_sha256:
            return path.resolve()
    raise RuntimeError("the single pinned PatchCore backbone is not cached with the expected hash")


def _load_samples(dataset: Path) -> list[dict[str, Any]]:
    samples = []
    with (dataset / "samples.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(
                {
                    "sample_id": record["sample_id"],
                    "entity_id": record["entity_id"],
                    "split": record["metadata"]["split"],
                    "result": record["response"]["result"],
                    "image_path": dataset / record["image"],
                    "mask_path": dataset / record["metadata"]["mask"],
                }
            )
    return sorted(samples, key=lambda item: item["sample_id"])


def _infer_tiles(
    model: Any,
    torch: Any,
    device: Any,
    samples: Sequence[dict[str, Any]],
    config: TiledPatchcoreConfig,
) -> list[dict[str, Any]]:
    outputs = []
    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            offsets = tile_offsets(height, config.tile_size)
            tile_maps = []
            tile_scores = []
            started = time.perf_counter()
            for offset in offsets:
                tensor = _tile_crop(rgb, offset, config.tile_size, config.tile_resize)
                tensor = tensor.unsqueeze(0).to(device)
                with torch.inference_mode():
                    prediction = model(tensor)
                tile_maps.append(
                    normalize_heatmap(
                        prediction.anomaly_map.detach().cpu().numpy().squeeze().astype(np.float32)
                    )
                )
                tile_scores.append(float(prediction.pred_score.detach().cpu().reshape(-1)[0]))
                del tensor, prediction
            if device.type == "mps":
                torch.mps.synchronize()
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        outputs.append(
            {
                "sample": sample,
                "width": width,
                "height": height,
                "offsets": offsets,
                "tile_maps": tile_maps,
                "score": max(tile_scores),
                "latency_ms": latency_ms,
            }
        )
    return outputs


def run_tiled_patchcore(
    config_path: str | Path = DEFAULT_CONFIG,
    dataset_root: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = load_tiled_config(config_path)
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tiled run: {output}")
    output.mkdir(parents=True)
    if _sha256(dataset / "manifest.json") != config.dataset_manifest_sha256:
        raise ValueError("KSDD manifest hash differs from tiled specialist config")
    samples = _load_samples(dataset)
    train_normal = [s for s in samples if s["split"] == "train" and s["result"] == "compliant"]
    validation = [s for s in samples if s["split"] == "validation"]
    test = [s for s in samples if s["split"] == "test"]
    if (len(train_normal), len(validation), len(test)) != (
        config.expected_train_normal,
        config.expected_validation,
        config.expected_test,
    ):
        raise ValueError("frozen KSDD populations differ from tiled specialist config")
    backbone_path = _verify_backbone(config)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from anomalib.models import Patchcore

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    device = torch.device(config.device)
    if device.type == "mps":
        torch.mps.empty_cache()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    module = Patchcore(
        backbone=config.backbone,
        layers=config.layers,
        pre_trained=True,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        num_neighbors=config.num_neighbors,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    model = module.model.to(device)
    model.train()
    feature_started = time.perf_counter()
    peak_mps_bytes = 0
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
        peak_mps_bytes = max(peak_mps_bytes, _mps_driver_memory(torch))
        del embedding, batch, tensors
    feature_seconds = time.perf_counter() - feature_started
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    coreset_started = time.perf_counter()
    module.fit()
    coreset_seconds = time.perf_counter() - coreset_started
    memory_bank_rows = int(model.memory_bank.shape[0])

    checkpoint_path = output / "patchcore_tiled_resnet18.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "source_manifest_sha256": config.dataset_manifest_sha256,
        },
        checkpoint_path,
    )
    model.to(device).eval()

    validation_outputs = _infer_tiles(model, torch, device, validation, config)
    image_threshold, image_calibration = select_classification_threshold(
        [item["score"] for item in validation_outputs],
        [item["sample"]["result"] == "violation" for item in validation_outputs],
    )
    test_outputs = _infer_tiles(model, torch, device, test, config)

    scores_path = output / "scores.jsonl"
    heatmap_root = output / "heatmaps"
    latencies = []
    with scores_path.open("w", encoding="utf-8") as stream:
        for split, split_outputs in (("validation", validation_outputs), ("test", test_outputs)):
            for item in split_outputs:
                sample = item["sample"]
                record: dict[str, Any] = {
                    "split": split,
                    "sample_id": sample["sample_id"],
                    "entity_id": sample["entity_id"],
                    "result": sample["result"],
                    "image_width": item["width"],
                    "image_height": item["height"],
                    "anomaly_score": item["score"],
                    "tile_offsets": list(item["offsets"]),
                    "latency_ms": item["latency_ms"],
                    "fusion": {},
                }
                for mode in config.fusion_modes:
                    fused = _fuse_tile_heatmaps(
                        item["tile_maps"],
                        item["offsets"],
                        (item["width"], item["height"]),
                        config.tile_size,
                        mode,
                    )
                    heatmap_path = heatmap_root / mode / f"{sample['sample_id']}.npy"
                    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(heatmap_path, fused.astype(np.float32))
                    record["fusion"][mode] = {
                        "heatmap_npy": str(heatmap_path),
                        "heatmap_sha256": _sha256(heatmap_path),
                        "heatmap_shape": list(fused.shape),
                    }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                latencies.append(item["latency_ms"])

    ordered = sorted(latencies)
    report = {
        "schema_version": "1.0",
        "phase": "phase8",
        "kind": "patchcore_tiled",
        "status": "completed",
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config.experiment_id,
        "implementation": {
            "name": "Anomalib PatchCore (tiled)",
            "algorithm": config.algorithm,
            "license": "Apache-2.0",
        },
        "dataset": {
            "version": config.dataset_version,
            "manifest_sha256": config.dataset_manifest_sha256,
            "train_normal": len(train_normal),
            "validation": len(validation),
            "test": len(test),
            "license": "CC-BY-NC-SA-4.0",
        },
        "backbone": {
            "name": config.backbone,
            "repository": config.backbone_repository,
            "revision": config.backbone_revision,
            "sha256": config.backbone_sha256,
            "bytes": config.backbone_bytes,
            "cache_path": str(backbone_path),
            "additional_model_downloads": 0,
        },
        "configuration": asdict(config),
        "fit": {
            "feature_extraction_seconds": round(feature_seconds, 3),
            "coreset_seconds": round(coreset_seconds, 3),
            "memory_bank_rows": memory_bank_rows,
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "calibration": {
            "split": "validation",
            "image_threshold": image_threshold,
            "image_metrics": image_calibration,
            "test_labels_used": False,
        },
        "performance": {
            "images": len(latencies),
            "p50_latency_ms": ordered[max(1, math.ceil(0.50 * len(ordered))) - 1],
            "p95_latency_ms": ordered[max(1, math.ceil(0.95 * len(ordered))) - 1],
            "process_peak_rss_mb": process_peak_memory_mb(),
            "mps_driver_peak_observed_mb": round(peak_mps_bytes / (1024 * 1024), 3),
            "wall_seconds": round(time.perf_counter() - started, 3),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "scores": str(scores_path),
            "scores_sha256": _sha256(scores_path),
        },
    }
    write_json_atomic(report, output / "run_manifest.json")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_tiled_patchcore(args.config, args.dataset_root, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "memory_bank_rows": report["fit"]["memory_bank_rows"],
        "image_threshold": report["calibration"]["image_threshold"],
        "validation_macro_f1": report["calibration"]["image_metrics"]["macro_f1"],
        "p50_latency_ms": report["performance"]["p50_latency_ms"],
        "p95_latency_ms": report["performance"]["p95_latency_ms"],
        "process_peak_rss_mb": report["performance"]["process_peak_rss_mb"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
