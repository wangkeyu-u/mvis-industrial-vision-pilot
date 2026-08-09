"""Phase 8: re-infer frozen PatchCore checkpoint and dump lossless float heatmaps.

Phase 7 persisted only uint8 PNG heatmaps. Failure analysis and post-processing
experiments need the raw float32 anomaly maps at model resolution, so this
module reloads the hash-pinned checkpoint through the same validation path as
the service bridge and re-runs validation + test inference deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .service_bridge import create_specialist_service_adapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "artifacts/model/phase7/patchcore_resnet18/run_manifest.json"
DEFAULT_DATASET = ROOT / "data/processed/ksdd_v0"
DEFAULT_OUTPUT = ROOT / "artifacts/model/phase8/patchcore_reinfer"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_samples(dataset: Path, split: str) -> list[dict[str, Any]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    samples = []
    with (dataset / "samples.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["metadata"]["split"] != split:
                continue
            samples.append(
                {
                    "sample_id": record["sample_id"],
                    "entity_id": record["entity_id"],
                    "result": record["response"]["result"],
                    "image_path": dataset / record["image"],
                    "mask_path": dataset / record["metadata"]["mask"],
                }
            )
    return sorted(samples, key=lambda item: item["sample_id"])


def reinfer_split(
    bridge: Any,
    samples: Sequence[dict[str, Any]],
    heatmap_dir: Path,
) -> list[dict[str, Any]]:
    """Run the pinned model and persist float32 heatmaps plus per-sample scores."""

    import torch

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            resized = rgb.resize(
                (bridge._image_size, bridge._image_size), Image.Resampling.BILINEAR
            )
            values = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(bridge._device)
        with torch.inference_mode():
            prediction = bridge._model(tensor)
            if bridge._device.type == "mps":
                torch.mps.synchronize()
        score = float(prediction.pred_score.detach().cpu().reshape(-1)[0])
        raw_map = prediction.anomaly_map.detach().cpu().numpy().squeeze().astype(np.float32)
        if raw_map.ndim != 2:
            raise ValueError(f"unexpected anomaly map shape {raw_map.shape}")
        heatmap_path = heatmap_dir / f"{sample['sample_id']}.npy"
        np.save(heatmap_path, raw_map)
        records.append(
            {
                "sample_id": sample["sample_id"],
                "entity_id": sample["entity_id"],
                "result": sample["result"],
                "image_width": width,
                "image_height": height,
                "anomaly_score": score,
                "heatmap_shape": list(raw_map.shape),
                "heatmap_npy": str(heatmap_path),
                "heatmap_sha256": _sha256(heatmap_path),
            }
        )
        del tensor, prediction
    return records


def run_reinfer(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    dataset_root: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT,
    splits: Sequence[str] = ("validation", "test"),
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reinfer output: {output}")
    output.mkdir(parents=True)
    dataset = Path(dataset_root).resolve()
    bridge = create_specialist_service_adapter(manifest_path)
    if not bridge.ready:
        raise RuntimeError("PatchCore bridge failed to load")

    started_at = datetime.now(timezone.utc).isoformat()
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        samples = load_split_samples(dataset, split)
        split_records[split] = reinfer_split(bridge, samples, output / "heatmaps" / split)
    scores_path = output / "scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as stream:
        for split in splits:
            for record in split_records[split]:
                stream.write(
                    json.dumps({"split": split, **record}, sort_keys=True) + "\n"
                )
    manifest = {
        "schema_version": "1.0",
        "phase": "phase8",
        "kind": "patchcore_reinfer",
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "source_run_manifest": str(Path(manifest_path).resolve()),
        "checkpoint_sha256": bridge._checkpoint_sha256,
        "image_threshold": bridge._image_threshold,
        "bbox_threshold": bridge._bbox_threshold,
        "image_size": bridge._image_size,
        "dataset_root": str(dataset),
        "splits": {split: len(split_records[split]) for split in splits},
        "scores_jsonl": str(scores_path),
        "scores_jsonl_sha256": _sha256(scores_path),
        "test_labels_used": False,
    }
    manifest_path_out = output / "run_manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = run_reinfer(args.manifest, args.dataset_root, args.output_dir)
    print(json.dumps({"status": "completed", "splits": manifest["splits"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
