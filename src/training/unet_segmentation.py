"""Phase 8 step 3: lightweight supervised U-Net segmentation baseline.

A fair supervised counterpart to PatchCore: same entity-isolated splits, same
validation-only parameter selection, same original-image evaluation harness.
The encoder is the same pinned timm ResNet-18 backbone (loaded from the
verified local cache; no downloads). Training runs on Apple Silicon MPS in
fp32 well under the 12 GB process gate.
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

from src.inference.patchcore_specialist import _sha256, select_classification_threshold
from src.inference.patchcore_tiled import _verify_backbone
from src.inference.performance import process_peak_memory_mb
from src.inference.real_probe import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/models/ksdd_unet_resnet18_phase8.json"
DEFAULT_TILES = ROOT / "data/processed/ksdd_tiled_v1"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8/unet_resnet18"


@dataclass(frozen=True, slots=True)
class UnetConfig:
    schema_version: str
    experiment_id: str
    encoder: str
    encoder_repository: str
    encoder_revision: str
    encoder_sha256: str
    encoder_bytes: int
    input_size: int
    tile_size: int
    tile_stride: int
    batch_size: int
    max_epochs: int
    early_stop_patience: int
    learning_rate: float
    weight_decay: float
    bce_weight: float
    dice_weight: float
    seed: int
    device: str
    dataset_version: str
    dataset_manifest_sha256: str
    tile_manifest_sha256: str
    expected_train_tiles: int
    expected_validation_tiles: int
    expected_test_tiles: int
    fusion_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported U-Net config schema_version")
        if len(self.encoder_sha256) != 64 or len(self.dataset_manifest_sha256) != 64:
            raise ValueError("configured hashes must be SHA-256")
        if self.input_size <= 0 or self.tile_size <= 0 or self.tile_stride <= 0:
            raise ValueError("input, tile, and stride must be positive")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.early_stop_patience <= 0:
            raise ValueError("batch, epochs, and patience must be positive")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError("learning rate must be in (0, 1)")
        if self.bce_weight <= 0.0 or self.dice_weight <= 0.0:
            raise ValueError("loss weights must be positive")
        unknown = set(self.fusion_modes) - {"max", "mean", "weighted"}
        if unknown or not self.fusion_modes:
            raise ValueError(f"unsupported fusion modes: {sorted(unknown)}")


def load_unet_config(path: str | Path) -> UnetConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = set(UnetConfig.__dataclass_fields__)
    if set(payload) != required:
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        raise ValueError(f"U-Net config fields differ; missing={missing}, unknown={unknown}")
    payload["fusion_modes"] = tuple(payload["fusion_modes"])
    return UnetConfig(**payload)


def build_unet(torch: Any, encoder_weights: Path) -> Any:
    """Compact U-Net: timm ResNet-18 feature encoder + light skip decoder."""

    nn = torch.nn

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class Unet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            import timm

            self.encoder = timm.create_model(
                "resnet18.a1_in1k", pretrained=False, features_only=True
            )
            # timm resnet18 features_only strides 2/4/8/16/32, channels 64/64/128/256/512
            channels = [64, 64, 128, 256, 512]
            self.decoders = nn.ModuleList(
                [
                    ConvBlock(channels[4] + channels[3], 256),
                    ConvBlock(256 + channels[2], 128),
                    ConvBlock(128 + channels[1], 64),
                    ConvBlock(64 + channels[0], 32),
                ]
            )
            self.head = nn.Conv2d(32, 1, 1)

        def forward(self, x: Any) -> Any:
            features = self.encoder(x)
            x = features[-1].contiguous()
            for index, decoder in enumerate(self.decoders):
                x = torch.nn.functional.interpolate(
                    x, scale_factor=2.0, mode="bilinear", align_corners=False
                ).contiguous()
                skip = features[-2 - index].contiguous()
                if x.shape[-2:] != skip.shape[-2:]:
                    x = torch.nn.functional.interpolate(
                        x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                    ).contiguous()
                x = decoder(torch.cat([x, skip], dim=1).contiguous())
            logits = self.head(x)
            return torch.nn.functional.interpolate(
                logits, size=(256, 256), mode="bilinear", align_corners=False
            ).contiguous()

    model = Unet()
    from safetensors.torch import load_file

    weights = load_file(str(encoder_weights))
    model.encoder.load_state_dict(weights, strict=False)
    return model


def _dice_loss(torch: Any, logits: Any, targets: Any) -> Any:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def _load_tile_records(tiles_root: Path, tile_size: int) -> list[dict[str, Any]]:
    records = []
    with (tiles_root / "tile_records.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["tile_size"] != tile_size:
                continue
            records.append(
                {
                    "tile_id": record["tile_id"],
                    "split": record["split"],
                    "label": record["label"],
                    "entity_id": record["entity_id"],
                    "image_path": tiles_root / record["image"],
                    "mask_path": (tiles_root / record["mask"]) if record["mask"] else None,
                }
            )
    return sorted(records, key=lambda item: item["tile_id"])


def _tile_arrays(record: dict[str, Any], input_size: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(record["image_path"]) as image:
        rgb = image.convert("RGB")
        if rgb.size != (input_size, input_size):
            rgb = rgb.resize((input_size, input_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(rgb, dtype=np.float32) / 255.0
    if record["mask_path"] is not None:
        with Image.open(record["mask_path"]) as mask_image:
            mask = mask_image.resize(
                (input_size, input_size), Image.Resampling.NEAREST
            )
            mask_array = (np.asarray(mask) > 0).astype(np.float32)
    else:
        mask_array = np.zeros((input_size, input_size), dtype=np.float32)
    return image_array, mask_array


def _normalize_batch(torch: Any, images: np.ndarray) -> Any:
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((tensor - mean) / std).contiguous()


def _validation_tile_dice(
    model: Any, torch: Any, device: Any, records: Sequence[dict[str, Any]], input_size: int
) -> float:
    model.eval()
    intersection = union = 0.0
    with torch.inference_mode():
        for record in records:
            image_array, mask_array = _tile_arrays(record, input_size)
            batch = _normalize_batch(torch, image_array[None]).to(device)
            probs = torch.sigmoid(model(batch)).detach().cpu().numpy()[0, 0]
            predicted = probs >= 0.5
            target = mask_array > 0
            intersection += float(np.logical_and(predicted, target).sum())
            union += float(predicted.sum() + target.sum())
    model.train()
    return (2.0 * intersection + 1.0) / (union + 1.0)


def _grid_offsets(length: int, tile: int, stride: int) -> tuple[int, ...]:
    if length <= tile:
        return (0,)
    offsets = list(range(0, length - tile + 1, stride))
    if offsets[-1] != length - tile:
        offsets.append(length - tile)
    return tuple(sorted(set(offsets)))


def _infer_images(
    model: Any,
    torch: Any,
    device: Any,
    samples: Sequence[dict[str, Any]],
    config: UnetConfig,
) -> list[dict[str, Any]]:
    outputs = []
    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            x_offsets = _grid_offsets(width, config.tile_size, config.tile_stride)
            y_offsets = _grid_offsets(height, config.tile_size, config.tile_stride)
            started = time.perf_counter()
            tiles = []
            positions = []
            with torch.inference_mode():
                for y_offset in y_offsets:
                    for x_offset in x_offsets:
                        crop = rgb.crop(
                            (x_offset, y_offset, x_offset + config.tile_size, y_offset + config.tile_size)
                        )
                        if config.tile_size != config.input_size:
                            crop = crop.resize(
                                (config.input_size, config.input_size), Image.Resampling.BILINEAR
                            )
                        array = np.asarray(crop, dtype=np.float32) / 255.0
                        batch = _normalize_batch(torch, array[None]).to(device)
                        probs = torch.sigmoid(model(batch)).detach().cpu().numpy()[0, 0]
                        tiles.append(probs.astype(np.float32))
                        positions.append((x_offset, y_offset))
            if device.type == "mps":
                torch.mps.synchronize()
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        outputs.append(
            {
                "sample": sample,
                "width": width,
                "height": height,
                "tiles": tiles,
                "positions": positions,
                "score": float(max(tile.max() for tile in tiles)),
                "latency_ms": latency_ms,
            }
        )
    return outputs


def _fuse_grid(
    item: dict[str, Any], config: UnetConfig, mode: str
) -> np.ndarray:
    """Fuse per-tile probability maps into one native-resolution heatmap."""

    import cv2

    width, height = item["width"], item["height"]
    accumulator = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for tile, (x_offset, y_offset) in zip(item["tiles"], item["positions"]):
        upsampled = cv2.resize(
            tile, (config.tile_size, config.tile_size), interpolation=cv2.INTER_LINEAR
        )
        if mode == "max":
            region = accumulator[
                y_offset : y_offset + config.tile_size, x_offset : x_offset + config.tile_size
            ]
            np.maximum(region, upsampled, out=region)
            continue
        if mode == "mean":
            weights = np.ones((config.tile_size, config.tile_size), dtype=np.float32)
        elif mode == "weighted":
            center = config.tile_size / 2.0
            axis = 1.0 - np.abs(np.arange(config.tile_size) - center) / center + 1e-3
            weights = (axis[:, None] * axis[None, :]).astype(np.float32)
        else:
            raise ValueError(f"unsupported fusion mode: {mode}")
        accumulator[
            y_offset : y_offset + config.tile_size, x_offset : x_offset + config.tile_size
        ] += upsampled * weights
        weight_sum[
            y_offset : y_offset + config.tile_size, x_offset : x_offset + config.tile_size
        ] += weights
    if mode == "max":
        return accumulator
    return accumulator / np.maximum(weight_sum, 1e-6)


def run_unet_training(
    config_path: str | Path = DEFAULT_CONFIG,
    tiles_root: str | Path = DEFAULT_TILES,
    dataset_root: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = load_unet_config(config_path)
    tiles_root = Path(tiles_root).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite U-Net run: {output}")
    output.mkdir(parents=True)
    if _sha256(dataset / "manifest.json") != config.dataset_manifest_sha256:
        raise ValueError("KSDD manifest hash differs from U-Net config")
    tile_manifest_sha = _sha256(tiles_root / "tile_manifest.json")
    if tile_manifest_sha != config.tile_manifest_sha256:
        raise ValueError("tile manifest hash differs from U-Net config")
    encoder_weights = _verify_encoder(config)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    device = torch.device(config.device)

    records = _load_tile_records(tiles_root, config.tile_size)
    train_records = [r for r in records if r["split"] == "train"]
    validation_records = [r for r in records if r["split"] == "validation"]
    test_records = [r for r in records if r["split"] == "test"]
    if (len(train_records), len(validation_records), len(test_records)) != (
        config.expected_train_tiles,
        config.expected_validation_tiles,
        config.expected_test_tiles,
    ):
        raise ValueError("frozen tile populations differ from U-Net config")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    model = build_unet(torch, encoder_weights).to(device)
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
    history = []
    peak_mps_bytes = 0
    checkpoint_path = output / "unet_resnet18.pt"
    stopped = False
    for epoch in range(config.max_epochs):
        rng = random.Random(config.seed + epoch)
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
        rng.shuffle(epoch_order)
        epoch_loss = 0.0
        steps = 0
        for batch_start in range(0, len(epoch_order), config.batch_size):
            batch_indices = epoch_order[batch_start : batch_start + config.batch_size]
            images = []
            masks = []
            for index in batch_indices:
                image_array, mask_array = _tile_arrays(train_records[index], config.input_size)
                images.append(image_array)
                masks.append(mask_array)
            batch = _normalize_batch(torch, np.stack(images)).to(device)
            targets = torch.from_numpy(np.stack(masks)[:, None]).to(device)
            logits = model(batch)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            dice = _dice_loss(torch, logits, targets)
            loss = config.bce_weight * bce + config.dice_weight * dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
            peak_mps_bytes = max(
                peak_mps_bytes,
                int(torch.mps.driver_allocated_memory()) if device.type == "mps" else 0,
            )
        validation_dice = _validation_tile_dice(
            model, torch, device, validation_records, config.input_size
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_loss / max(steps, 1), 6),
                "validation_tile_dice": round(validation_dice, 6),
            }
        )
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "source_manifest_sha256": config.dataset_manifest_sha256,
                    "tile_manifest_sha256": tile_manifest_sha,
                    "best_epoch": best_epoch,
                    "validation_tile_dice": best_dice,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                stopped = True
                break
    train_seconds = time.perf_counter() - started

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    samples = _load_source_samples(dataset)
    validation_samples = [s for s in samples if s["split"] == "validation"]
    test_samples = [s for s in samples if s["split"] == "test"]
    validation_outputs = _infer_images(model, torch, device, validation_samples, config)
    image_threshold, image_calibration = select_classification_threshold(
        [item["score"] for item in validation_outputs],
        [item["sample"]["result"] == "violation" for item in validation_outputs],
    )
    test_outputs = _infer_images(model, torch, device, test_samples, config)

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
                    "latency_ms": item["latency_ms"],
                    "fusion": {},
                }
                for mode in config.fusion_modes:
                    fused = _fuse_grid(item, config, mode)
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
        "kind": "unet_supervised_segmentation",
        "status": "completed",
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config.experiment_id,
        "dataset": {
            "version": config.dataset_version,
            "manifest_sha256": config.dataset_manifest_sha256,
            "tile_manifest_sha256": tile_manifest_sha,
            "train_tiles": len(train_records),
            "validation_tiles": len(validation_records),
            "test_tiles": len(test_records),
            "train_positive_tiles": positive_count,
            "license": "CC-BY-NC-SA-4.0",
        },
        "encoder": {
            "name": config.encoder,
            "repository": config.encoder_repository,
            "revision": config.encoder_revision,
            "sha256": config.encoder_sha256,
            "cache_path": str(encoder_weights),
            "additional_model_downloads": 0,
        },
        "configuration": asdict(config),
        "training": {
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_validation_tile_dice": best_dice,
            "early_stopped": stopped,
            "train_seconds": round(train_seconds, 3),
            "history": history,
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


def _verify_encoder(config: UnetConfig) -> Path:
    class _Adapter:
        backbone_revision = config.encoder_revision
        backbone_bytes = config.encoder_bytes
        backbone_sha256 = config.encoder_sha256

    return _verify_backbone(_Adapter())


def _load_source_samples(dataset: Path) -> list[dict[str, Any]]:
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
                }
            )
    return sorted(samples, key=lambda item: item["sample_id"])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tiles-root", type=Path, default=DEFAULT_TILES)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_unet_training(args.config, args.tiles_root, args.dataset_root, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "epochs_ran": report["training"]["epochs_ran"],
        "best_validation_tile_dice": report["training"]["best_validation_tile_dice"],
        "image_threshold": report["calibration"]["image_threshold"],
        "validation_macro_f1": report["calibration"]["image_metrics"]["macro_f1"],
        "p50_latency_ms": report["performance"]["p50_latency_ms"],
        "process_peak_rss_mb": report["performance"]["process_peak_rss_mb"],
        "mps_driver_peak_observed_mb": report["performance"]["mps_driver_peak_observed_mb"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
