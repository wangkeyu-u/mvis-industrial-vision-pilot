"""Phase 8: render original-image overlays for a selected specialist model.

Side-by-side panels: original image with truth boxes (green) and predicted
boxes (red), plus the fused heatmap with the same boxes. Used for the phase 8
failure/success case reports; reads only frozen artifacts, never tunes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from src.evaluation.localization_postprocess import (
    PostprocessConfig,
    extract_boxes,
    intersection_over_union,
    scale_box_to_original,
    truth_boxes_from_mask,
)
from src.inference.patchcore_reinfer import load_split_samples

ROOT = Path(__file__).resolve().parents[2]


def render_overlays(
    *,
    scores_path: Path,
    split: str,
    fusion_mode: str,
    config: PostprocessConfig,
    dataset_root: Path,
    output_dir: Path,
    only_positives: bool = True,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = {s["sample_id"]: s for s in load_split_samples(dataset_root, split)}
    rendered = []
    with scores_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["split"] != split:
                continue
            sample_id = record["sample_id"]
            sample = samples[sample_id]
            if only_positives and sample["result"] != "violation":
                continue
            fusion = record["fusion"][fusion_mode]
            heatmap = np.load(fusion["heatmap_npy"])
            with Image.open(sample["image_path"]) as image:
                base = image.convert("RGB")
                width, height = base.size
            with Image.open(sample["mask_path"]) as mask_image:
                truth_mask = np.asarray(mask_image)
            truth_boxes = truth_boxes_from_mask(truth_mask)
            boxes, _, _ = extract_boxes(heatmap, config)
            scaled = [
                scale_box_to_original(box, heatmap.shape, (width, height)) for box in boxes
            ]
            maximum = float(heatmap.max())
            normalized = heatmap / maximum if maximum > 0 else heatmap
            heat_image = Image.fromarray(
                np.uint8(np.clip(normalized * 255.0, 0, 255)), mode="L"
            ).convert("RGB")
            for canvas in (base, heat_image):
                draw = ImageDraw.Draw(canvas)
                for box in truth_boxes:
                    draw.rectangle(list(box), outline=(0, 200, 0), width=2)
                for box in scaled:
                    draw.rectangle(list(box), outline=(220, 0, 0), width=2)
            combined = Image.new("RGB", (width * 2, height))
            combined.paste(base, (0, 0))
            combined.paste(heat_image, (width, 0))
            combined.save(output_dir / f"{sample_id}.png")
            rendered.append(
                {
                    "sample_id": sample_id,
                    "score": record["anomaly_score"],
                    "predicted_boxes": [list(box) for box in scaled],
                    "truth_boxes": [list(box) for box in truth_boxes],
                    "best_iou": max(
                        (
                            intersection_over_union(prediction, truth)
                            for prediction in scaled
                            for truth in truth_boxes
                        ),
                        default=0.0,
                    ),
                }
            )
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fusion-mode", type=str, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "data/processed/ksdd_v0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    config = PostprocessConfig(**json.loads(args.config_json.read_text(encoding="utf-8")))
    rendered = render_overlays(
        scores_path=args.scores,
        split=args.split,
        fusion_mode=args.fusion_mode,
        config=config,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        only_positives=not args.all,
    )
    print(json.dumps({"rendered": len(rendered)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
