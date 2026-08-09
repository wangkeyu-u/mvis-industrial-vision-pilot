"""Offline evaluation CLI for tiled Anomalib or project specialist outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from src.data.ksdd_tiles import load_tile_records

from .specialist import (
    SpecialistTileCase,
    evaluate_specialist_tiles,
    normalize_specialist_prediction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-records", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--heatmap-threshold", type=float, required=True)
    parser.add_argument("--image-threshold", type=float, required=True)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--box-iou-threshold", type=float, default=0.5)
    return parser


def _load_predictions(path: str | Path) -> dict[str, Mapping[str, Any]]:
    rows = load_tile_records(path)
    values = {}
    for row in rows:
        normalized = normalize_specialist_prediction(row)
        tile_id = normalized["tile_id"]
        if not tile_id:
            raise ValueError("every specialist prediction requires tile_id")
        if tile_id in values:
            raise ValueError(f"duplicate specialist prediction: {tile_id}")
        values[tile_id] = normalized
    return values


def _mask_grid(record: Mapping[str, Any], root: Path) -> tuple[tuple[int, ...], ...]:
    mask_value = record.get("mask")
    if isinstance(mask_value, str):
        path = root / mask_value
        with Image.open(path) as opened:
            mask = opened.convert("L")
            width, height = mask.size
            pixels = mask.tobytes()
        return tuple(
            tuple(int(bool(value)) for value in pixels[y * width : (y + 1) * width])
            for y in range(height)
        )
    size = int(record["tile_size"])
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))


def evaluate_prediction_file(
    tile_records_path: str | Path,
    dataset_root: str | Path,
    predictions_path: str | Path,
    *,
    split: str,
    heatmap_threshold: float,
    image_threshold: float,
    min_component_area: int = 1,
    box_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    records = {
        str(row["tile_id"]): row
        for row in load_tile_records(tile_records_path)
        if row.get("split") == split
    }
    predictions = _load_predictions(predictions_path)
    missing = sorted(set(records) - set(predictions))
    extra = sorted(set(predictions) - set(records))
    if missing or extra:
        raise ValueError(f"prediction IDs must exactly match split; missing={missing}, extra={extra}")
    root = Path(dataset_root).resolve()
    cases = []
    for tile_id in sorted(records):
        record, prediction = records[tile_id], predictions[tile_id]
        declared_source = prediction.get("source_sample_id")
        if declared_source and declared_source != record["source_sample_id"]:
            raise ValueError(f"source_sample_id mismatch for {tile_id}")
        cases.append(
            SpecialistTileCase(
                tile_id=tile_id,
                source_sample_id=str(record["source_sample_id"]),
                anomaly_map=prediction["anomaly_map"],
                target_mask=_mask_grid(record, root),
                image_score=float(prediction["image_score"]),
                transform=record["transform"],
            )
        )
    report = evaluate_specialist_tiles(
        cases,
        heatmap_threshold=heatmap_threshold,
        image_threshold=image_threshold,
        min_component_area=min_component_area,
        box_iou_threshold=box_iou_threshold,
    )
    report["split"] = split
    report["prediction_count"] = len(predictions)
    report["fixture_or_mock"] = False
    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = evaluate_prediction_file(
        arguments.tile_records,
        arguments.dataset_root,
        arguments.predictions,
        split=arguments.split,
        heatmap_threshold=arguments.heatmap_threshold,
        image_threshold=arguments.image_threshold,
        min_component_area=arguments.min_component_area,
        box_iou_threshold=arguments.box_iou_threshold,
    )
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(destination),
                "prediction_count": report["prediction_count"],
                "pilot_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
