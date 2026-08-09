"""Official Anomalib PatchCore runner for the frozen KSDD V0 split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .performance import process_peak_memory_mb
from .real_probe import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/models/ksdd_patchcore_resnet18_phase7.json"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase7/patchcore_resnet18"


@dataclass(frozen=True, slots=True)
class PatchcoreConfig:
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
    image_size: int
    train_batch_size: int
    coreset_sampling_ratio: float
    num_neighbors: int
    seed: int
    device: str
    dataset_version: str
    dataset_manifest_sha256: str
    expected_train_normal: int
    expected_validation: int
    expected_test: int
    bbox_threshold_candidates: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported PatchCore config schema_version")
        if self.implementation != "anomalib" or self.algorithm != "patchcore":
            raise ValueError("phase 7 specialist must use official Anomalib PatchCore")
        if self.backbone_bytes <= 0 or self.backbone_bytes > 3_000_000_000:
            raise ValueError("backbone size violates the 3GB download gate")
        if len(self.backbone_sha256) != 64 or len(self.dataset_manifest_sha256) != 64:
            raise ValueError("configured hashes must be SHA-256")
        if self.image_size <= 0 or self.train_batch_size <= 0 or self.num_neighbors <= 0:
            raise ValueError("image, batch, and neighbor values must be positive")
        if not 0.0 < self.coreset_sampling_ratio <= 1.0:
            raise ValueError("coreset_sampling_ratio must be in (0, 1]")
        if not self.layers or not self.bbox_threshold_candidates:
            raise ValueError("layers and bbox thresholds must not be empty")
        if any(not 0.0 < value < 1.0 for value in self.bbox_threshold_candidates):
            raise ValueError("bbox thresholds must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class SpecialistSample:
    sample_id: str
    split: str
    result: str
    image_path: Path
    mask_path: Path
    image_sha256: str
    image_width: int
    image_height: int


def load_patchcore_config(path: str | Path) -> PatchcoreConfig:
    payload = _read_json_object(Path(path))
    required = set(PatchcoreConfig.__dataclass_fields__)
    if set(payload) != required:
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        raise ValueError(f"PatchCore config fields differ; missing={missing}, unknown={unknown}")
    payload["layers"] = tuple(payload["layers"])
    payload["bbox_threshold_candidates"] = tuple(payload["bbox_threshold_candidates"])
    return PatchcoreConfig(**payload)


def select_classification_threshold(
    scores: Sequence[float], labels: Sequence[bool]
) -> tuple[float, dict[str, float]]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have equal non-zero length")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("classification scores must be finite")
    ordered = sorted(set(float(score) for score in scores))
    candidates = [math.nextafter(ordered[0], -math.inf)]
    candidates.extend((left + right) / 2.0 for left, right in zip(ordered, ordered[1:]))
    candidates.append(math.nextafter(ordered[-1], math.inf))
    best: tuple[float, float, float] | None = None
    best_metrics: dict[str, float] = {}
    for threshold in candidates:
        predicted = [score >= threshold for score in scores]
        tp = sum(prediction and label for prediction, label in zip(predicted, labels))
        tn = sum(not prediction and not label for prediction, label in zip(predicted, labels))
        fp = sum(prediction and not label for prediction, label in zip(predicted, labels))
        fn = sum(not prediction and label for prediction, label in zip(predicted, labels))
        f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
        macro_f1 = (f1_positive + f1_negative) / 2.0
        accuracy = (tp + tn) / len(labels)
        ranking = (macro_f1, accuracy, threshold)
        if best is None or ranking > best:
            best = ranking
            best_metrics = {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "true_positive": float(tp),
                "true_negative": float(tn),
                "false_positive": float(fp),
                "false_negative": float(fn),
            }
    assert best is not None
    return best[2], best_metrics


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("heatmap contains non-finite values")
    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def heatmap_to_bbox(
    normalized: np.ndarray,
    threshold: float,
    original_width: int,
    original_height: int,
) -> tuple[int, int, int, int]:
    import cv2

    if normalized.ndim != 2:
        raise ValueError("normalized heatmap must be two-dimensional")
    binary = (normalized >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        x2, y2 = x + width, y + height
    else:
        y, x = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
        x2, y2 = min(normalized.shape[1], x + 2), min(normalized.shape[0], y + 2)
        x, y = max(0, x - 1), max(0, y - 1)
    map_height, map_width = normalized.shape
    scaled_x1 = max(0, min(original_width - 1, math.floor(x * original_width / map_width)))
    scaled_y1 = max(0, min(original_height - 1, math.floor(y * original_height / map_height)))
    scaled_x2 = max(scaled_x1 + 1, min(original_width, math.ceil(x2 * original_width / map_width)))
    scaled_y2 = max(scaled_y1 + 1, min(original_height, math.ceil(y2 * original_height / map_height)))
    return scaled_x1, scaled_y1, scaled_x2, scaled_y2


def select_bbox_threshold(
    heatmaps: Sequence[np.ndarray],
    truth_boxes: Sequence[tuple[int, int, int, int]],
    sizes: Sequence[tuple[int, int]],
    candidates: Sequence[float],
) -> tuple[float, dict[str, float]]:
    if not heatmaps or not (len(heatmaps) == len(truth_boxes) == len(sizes)):
        raise ValueError("bbox calibration inputs must have equal non-zero length")
    best: tuple[float, float, float] | None = None
    metrics: dict[str, float] = {}
    for threshold in candidates:
        ious = []
        for heatmap, truth, (width, height) in zip(heatmaps, truth_boxes, sizes):
            predicted = heatmap_to_bbox(heatmap, threshold, width, height)
            ious.append(_iou(predicted, truth))
        acc = sum(value >= 0.5 for value in ious) / len(ious)
        mean_iou = statistics.fmean(ious)
        ranking = (acc, mean_iou, threshold)
        if best is None or ranking > best:
            best = ranking
            metrics = {"acc_at_iou_0_5": acc, "mean_iou": mean_iou}
    assert best is not None
    return best[2], metrics


def build_specialist_prediction(
    *,
    score: float,
    image_threshold: float,
    bbox: tuple[int, int, int, int] | None,
    confidence: float,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    positive = score >= image_threshold
    if positive and bbox is None:
        raise ValueError("positive specialist prediction requires a bbox")
    return {
        "result": "violation" if positive else "compliant",
        "objects": (
            [
                {
                    "label": "surface_defect",
                    "bbox": list(bbox),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "source": "patchcore",
                }
            ]
            if positive
            else []
        ),
        "reason": (
            f"PatchCore anomaly score {score:.6f} exceeds validation threshold {image_threshold:.6f}."
            if positive
            else f"PatchCore anomaly score {score:.6f} is below validation threshold {image_threshold:.6f}."
        ),
        "uncertain": False,
        "refusal": None,
        "provenance": {
            "model_id": model_id,
            "model_revision": revision,
            "backend": "anomalib_patchcore",
            "config_fingerprint": revision[:16],
            "adapter_id": None,
            "seed": 20260808,
            "deterministic": True,
            "extra": {},
        },
        "warnings": [],
    }


def run_patchcore_specialist(
    config_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_patchcore_config(config_path)
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite specialist run: {output}")
    output.mkdir(parents=True)
    manifest_path = dataset / "manifest.json"
    if _sha256(manifest_path) != config.dataset_manifest_sha256:
        raise ValueError("KSDD manifest hash differs from specialist config")
    samples = _load_samples(dataset)
    train_normal = tuple(
        item for item in samples if item.split == "train" and item.result == "compliant"
    )
    validation = tuple(item for item in samples if item.split == "validation")
    test = tuple(item for item in samples if item.split == "test")
    if (len(train_normal), len(validation), len(test)) != (
        config.expected_train_normal,
        config.expected_validation,
        config.expected_test,
    ):
        raise ValueError("frozen KSDD populations differ from specialist config")
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
    started_at = _now()
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
    for batch_start in range(0, len(train_normal), config.train_batch_size):
        batch_samples = train_normal[batch_start : batch_start + config.train_batch_size]
        tensor = torch.stack(
            [_image_tensor(sample.image_path, config.image_size) for sample in batch_samples]
        ).to(device)
        with torch.inference_mode():
            embedding = model(tensor)
        model.embedding_store[-1] = embedding.detach().cpu()
        peak_mps_bytes = max(peak_mps_bytes, _mps_driver_memory(torch))
        del embedding, tensor
    feature_seconds = time.perf_counter() - feature_started
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    coreset_started = time.perf_counter()
    module.fit()
    coreset_seconds = time.perf_counter() - coreset_started
    memory_bank_rows = int(model.memory_bank.shape[0])
    memory_bank_dimensions = int(model.memory_bank.shape[1])

    checkpoint_path = output / "patchcore_resnet18.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "source_manifest_sha256": config.dataset_manifest_sha256,
        },
        checkpoint_path,
    )
    model.to(device).eval()
    validation_outputs = _infer_samples(model, validation, config.image_size, torch, device)
    image_threshold, image_calibration = select_classification_threshold(
        [item["score"] for item in validation_outputs],
        [sample.result == "violation" for sample in validation],
    )
    positive_calibration = [
        (sample, prediction)
        for sample, prediction in zip(validation, validation_outputs)
        if sample.result == "violation"
    ]
    bbox_threshold, bbox_calibration = select_bbox_threshold(
        [item[1]["normalized_heatmap"] for item in positive_calibration],
        [_mask_bbox(item[0].mask_path) for item in positive_calibration],
        [(item[0].image_width, item[0].image_height) for item in positive_calibration],
        config.bbox_threshold_candidates,
    )

    test_outputs = _infer_samples(model, test, config.image_size, torch, device)
    heatmap_dir = output / "heatmaps"
    heatmap_dir.mkdir()
    predictions_path = output / "predictions.jsonl"
    raw_scores_path = output / "raw_scores.jsonl"
    latencies = []
    with predictions_path.open("w", encoding="utf-8") as predictions, raw_scores_path.open(
        "w", encoding="utf-8"
    ) as raw_scores:
        validation_min = min(item["score"] for item in validation_outputs)
        validation_max = max(item["score"] for item in validation_outputs)
        for sample, result in zip(test, test_outputs):
            heatmap = result["normalized_heatmap"]
            resized = Image.fromarray(np.uint8(np.clip(heatmap * 255.0, 0, 255)), mode="L")
            resized = resized.resize(
                (sample.image_width, sample.image_height), Image.Resampling.BILINEAR
            )
            heatmap_path = heatmap_dir / f"{sample.sample_id}.png"
            resized.save(heatmap_path)
            positive = result["score"] >= image_threshold
            bbox = (
                heatmap_to_bbox(
                    heatmap, bbox_threshold, sample.image_width, sample.image_height
                )
                if positive
                else None
            )
            confidence = _score_confidence(
                result["score"], validation_min, validation_max, image_threshold
            )
            payload = build_specialist_prediction(
                score=result["score"],
                image_threshold=image_threshold,
                bbox=bbox,
                confidence=confidence,
                model_id="anomalib/patchcore-resnet18-ksdd-v0",
                revision=_sha256(checkpoint_path),
            )
            predictions.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "source": "model_result",
                        "prediction": payload,
                        "status": "completed",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            raw_scores.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "anomaly_score": result["score"],
                        "classification_threshold": image_threshold,
                        "bbox_threshold": bbox_threshold,
                        "bbox": list(bbox) if bbox else None,
                        "heatmap": str(heatmap_path),
                        "latency_ms": result["latency_ms"],
                        "mps_driver_allocated_mb": result["mps_driver_allocated_mb"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            latencies.append(result["latency_ms"])

    fit_seconds = feature_seconds + coreset_seconds
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("anomalib", "torch", "torchvision", "timm")
    }
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "started_at": started_at,
        "ended_at": _now(),
        "experiment_id": config.experiment_id,
        "implementation": {
            "name": "Anomalib PatchCore",
            "algorithm": config.algorithm,
            "dependencies": dependencies,
            "license": "Apache-2.0",
            "source": "https://github.com/open-edge-platform/anomalib",
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
            "total_seconds": round(fit_seconds, 3),
            "memory_bank_rows": memory_bank_rows,
            "memory_bank_dimensions": memory_bank_dimensions,
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "calibration": {
            "split": "validation",
            "image_threshold": image_threshold,
            "image_metrics": image_calibration,
            "bbox_threshold": bbox_threshold,
            "bbox_metrics": bbox_calibration,
            "test_labels_used": False,
        },
        "performance": {
            "test_images": len(test_outputs),
            "p50_latency_ms": _nearest_rank(latencies, 50),
            "p95_latency_ms": _nearest_rank(latencies, 95),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "process_peak_rss_mb": process_peak_memory_mb(),
            "mps_driver_peak_observed_mb": round(
                max(
                    [peak_mps_bytes]
                    + [int(item["mps_driver_allocated_mb"] * 1024 * 1024) for item in test_outputs]
                )
                / (1024 * 1024),
                3,
            ),
            "wall_seconds": round(time.perf_counter() - started, 3),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "raw_scores": str(raw_scores_path),
            "raw_scores_sha256": _sha256(raw_scores_path),
            "heatmap_directory": str(heatmap_dir),
        },
        "python": sys.version.split()[0],
    }
    write_json_atomic(report, output / "run_manifest.json")
    return report


def _load_samples(dataset: Path) -> tuple[SpecialistSample, ...]:
    manifest = _read_json_object(dataset / "manifest.json")
    entries = {item["sample_id"]: item for item in manifest["entries"]}
    values = []
    with (dataset / "samples.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record["sample_id"]
            entry = entries[sample_id]
            image_path = (dataset / record["image"]).resolve()
            mask_path = (dataset / record["metadata"]["mask"]).resolve()
            if _sha256(image_path) != entry["sha256"]:
                raise ValueError(f"image hash mismatch: {sample_id}")
            with Image.open(image_path) as image:
                width, height = image.size
            values.append(
                SpecialistSample(
                    sample_id,
                    record["metadata"]["split"],
                    record["response"]["result"],
                    image_path,
                    mask_path,
                    entry["sha256"],
                    width,
                    height,
                )
            )
    return tuple(values)


def _verify_backbone(config: PatchcoreConfig) -> Path:
    root = Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshot = (
        root
        / "models--timm--resnet18.a1_in1k"
        / "snapshots"
        / config.backbone_revision
    )
    candidates = [item for item in snapshot.rglob("*") if item.is_file()]
    for path in candidates:
        if path.stat().st_size == config.backbone_bytes and _sha256(path) == config.backbone_sha256:
            return path.resolve()
    raise RuntimeError("the single pinned PatchCore backbone is not cached with the expected hash")


def _image_tensor(path: Path, size: int) -> Any:
    import torch

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        values = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def _infer_samples(model: Any, samples: Sequence[SpecialistSample], size: int, torch: Any, device: Any) -> list[dict[str, Any]]:
    results = []
    for sample in samples:
        tensor = _image_tensor(sample.image_path, size).unsqueeze(0).to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(tensor)
        if device.type == "mps":
            torch.mps.synchronize()
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        score = float(prediction.pred_score.detach().cpu().reshape(-1)[0])
        heatmap = prediction.anomaly_map.detach().cpu().numpy().squeeze()
        results.append(
            {
                "score": score,
                "normalized_heatmap": normalize_heatmap(heatmap),
                "latency_ms": latency_ms,
                "mps_driver_allocated_mb": round(
                    _mps_driver_memory(torch) / (1024 * 1024), 3
                ),
            }
        )
        del tensor, prediction
    return results


def _mps_driver_memory(torch: Any) -> int:
    try:
        return int(torch.mps.driver_allocated_memory())
    except (AttributeError, RuntimeError):
        return 0


def _mask_bbox(path: Path) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        values = np.asarray(image)
    y, x = np.nonzero(values)
    if not len(x):
        raise ValueError(f"positive calibration mask is empty: {path}")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def _score_confidence(score: float, minimum: float, maximum: float, threshold: float) -> float:
    if maximum <= minimum:
        return 0.5
    normalized = (score - minimum) / (maximum - minimum)
    threshold_normalized = (threshold - minimum) / (maximum - minimum)
    distance = abs(normalized - threshold_normalized)
    return max(0.5, min(0.999, 0.5 + distance / 2.0))


def _iou(left: Sequence[int], right: Sequence[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    report = run_patchcore_specialist(
        arguments.config, arguments.dataset_root, arguments.output_dir
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "fit_seconds": report["fit"]["total_seconds"],
                **report["performance"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
