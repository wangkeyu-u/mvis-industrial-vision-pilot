"""Create an auditable tiled runtime view from the frozen data-thread KSDD SFT package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/processed/ksdd_sft_v1"


def tile_positions(image_height: int, tile_size: int) -> tuple[int, ...]:
    if image_height < tile_size:
        raise ValueError("source image height must be at least tile_size")
    return tuple(sorted({0, (image_height - tile_size) // 2, image_height - tile_size}))


def transform_answer_for_tile(
    answer: Mapping[str, Any],
    *,
    y_offset: int,
    source_tile_size: int,
    output_size: int,
) -> dict[str, Any]:
    objects = []
    for item in answer.get("objects", []):
        x1, y1, x2, y2 = item["bbox"]
        intersection_y1 = max(y1, y_offset)
        intersection_y2 = min(y2, y_offset + source_tile_size)
        if intersection_y1 >= intersection_y2 or x1 >= source_tile_size or x2 <= 0:
            continue
        clipped = (
            max(0, x1),
            intersection_y1 - y_offset,
            min(source_tile_size, x2),
            intersection_y2 - y_offset,
        )
        scale = output_size / source_tile_size
        scaled = [
            max(0, min(output_size - 1, math.floor(clipped[0] * scale))),
            max(0, min(output_size - 1, math.floor(clipped[1] * scale))),
            max(1, min(output_size, math.ceil(clipped[2] * scale))),
            max(1, min(output_size, math.ceil(clipped[3] * scale))),
        ]
        if scaled[0] >= scaled[2] or scaled[1] >= scaled[3]:
            continue
        objects.append({**item, "bbox": scaled})
    positive = bool(objects)
    return {
        "objects": objects,
        "reason": (
            "A visible surface anomaly is present in this tile. / 当前图块存在可见表面异常。"
            if positive
            else "No visible surface defect is present in this tile. / 当前图块未发现可见表面缺陷。"
        ),
        "result": "violation" if positive else "compliant",
        "uncertain": False,
    }


def export_tile_sft(
    source_root: str | Path,
    output_root: str | Path,
    *,
    image_size: int,
    created_at: str,
) -> dict[str, Any]:
    if image_size not in {256, 384}:
        raise ValueError("phase 7 tile image_size must be 256 or 384")
    source = Path(source_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite tile SFT: {destination}")
    source_manifest_path = source / "sft_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("source_manifest_sha256") != (
        "fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a"
    ):
        raise ValueError("source SFT is not based on frozen KSDD V0")
    if source_manifest.get("statistics", {}).get("test_derived_training_sample_count") != 0:
        raise ValueError("source SFT reports test-derived training samples")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("created_at must include timezone")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    split_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    source_ids: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}
    tile_ids: set[str] = set()
    source_tile_size = 500
    try:
        (staging / "hf").mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            records = _load_jsonl(source / "hf" / f"{split}.jsonl")
            output_file = staging / "hf" / f"{split}.jsonl"
            with output_file.open("w", encoding="utf-8") as stream:
                for record in records:
                    source_id = str(record["sample_id"])
                    source_ids[split].add(source_id)
                    image_reference = record["images"][0]
                    source_image = (ROOT / image_reference).resolve()
                    with Image.open(source_image) as image:
                        rgb = image.convert("RGB")
                        width, height = rgb.size
                        if width != source_tile_size:
                            raise ValueError(
                                f"tile policy expects width 500, got {width}: {source_id}"
                            )
                        positions = tile_positions(height, source_tile_size)
                        assistant = json.loads(record["messages"][-1]["content"][0]["text"])
                        for index, y_offset in enumerate(positions):
                            tile_id = f"{source_id}__tile{index}_y{y_offset}_s{image_size}"
                            if tile_id in tile_ids:
                                raise ValueError(f"duplicate tile sample_id: {tile_id}")
                            tile_ids.add(tile_id)
                            relative_image = Path("images") / split / f"{tile_id}.png"
                            tile_path = staging / relative_image
                            tile_path.parent.mkdir(parents=True, exist_ok=True)
                            tile = rgb.crop(
                                (0, y_offset, source_tile_size, y_offset + source_tile_size)
                            ).resize((image_size, image_size), Image.Resampling.LANCZOS)
                            tile.save(tile_path)
                            answer = transform_answer_for_tile(
                                assistant,
                                y_offset=y_offset,
                                source_tile_size=source_tile_size,
                                output_size=image_size,
                            )
                            messages = json.loads(json.dumps(record["messages"]))
                            final_reference = (
                                destination / relative_image
                            ).relative_to(ROOT).as_posix()
                            messages[0]["content"][0]["image"] = final_reference
                            messages[0]["content"][1]["text"] += (
                                f" The supplied image is one {image_size}x{image_size} tile; "
                                "all bbox coordinates must use this tile's pixel space."
                            )
                            messages[-1]["content"][0]["text"] = json.dumps(
                                answer,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            tiled = {
                                **record,
                                "sample_id": tile_id,
                                "images": [final_reference],
                                "messages": messages,
                                "tile": {
                                    "source_sample_id": source_id,
                                    "source_image_sha256": _sha256(source_image),
                                    "source_width": width,
                                    "source_height": height,
                                    "source_bbox_xyxy": [
                                        0,
                                        y_offset,
                                        source_tile_size,
                                        y_offset + source_tile_size,
                                    ],
                                    "output_size": image_size,
                                },
                            }
                            stream.write(
                                json.dumps(tiled, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            split_counts[split] += 1
                            positive_counts[split] += answer["result"] == "violation"
                stream.flush()
                os.fsync(stream.fileno())

        if source_ids["train"] & source_ids["test"]:
            raise ValueError("tile SFT source IDs leak between train and test")
        file_hashes = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "1.0",
            "dataset_version": f"ksdd_tile_sft-{image_size}-1.0.0",
            "source_sft_manifest_sha256": _sha256(source_manifest_path),
            "source_manifest_sha256": source_manifest["source_manifest_sha256"],
            "created_at": created_at,
            "tile_policy": {
                "source_tile_size": source_tile_size,
                "output_size": image_size,
                "vertical_positions": "top,middle,bottom",
                "bbox_policy": "clip_to_tile_then_scale_outward",
            },
            "statistics": {
                "split_counts": dict(split_counts),
                "positive_counts": dict(positive_counts),
                "negative_counts": {
                    split: split_counts[split] - positive_counts[split]
                    for split in split_counts
                },
                "source_split_counts": {
                    split: len(values) for split, values in source_ids.items()
                },
                "test_derived_training_sample_count": 0,
                "formal_kpi_eligible": False,
            },
            "format": {
                "name": "mlx-vlm images/messages tiled runtime view",
                "train_file": "hf/train.jsonl",
                "validation_file": "hf/valid.jsonl",
                "test_file": "hf/test.jsonl",
            },
            "leakage": {
                "train_test_source_sample_overlap": [],
                "source_assignments_preserved": True,
                "test_derived_training_sample_count": 0,
                "passes": True,
            },
            "file_sha256": file_hashes,
        }
        _write_json(staging / "sft_manifest.json", manifest)
        manifest_hash = _sha256(staging / "sft_manifest.json")
        (staging / "sft_manifest.sha256").write_text(
            f"{manifest_hash}  sft_manifest.json\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest | {"manifest_sha256": manifest_hash}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("SFT JSONL rows must be objects")
                values.append(value)
    return values


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, choices=(256, 384), required=True)
    parser.add_argument("--created-at", default=datetime.now(timezone.utc).isoformat())
    arguments = parser.parse_args(argv)
    result = export_tile_sft(
        arguments.source_root,
        arguments.output_root,
        image_size=arguments.image_size,
        created_at=arguments.created_at,
    )
    print(
        json.dumps(
            {
                "dataset_version": result["dataset_version"],
                "manifest_sha256": result["manifest_sha256"],
                "statistics": result["statistics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
