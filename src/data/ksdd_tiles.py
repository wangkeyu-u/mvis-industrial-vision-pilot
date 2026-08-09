"""Deterministic tiled derivatives of the frozen KSDD V0 entity split.

The exporter never re-splits source images.  Every tile inherits its source
entity and split, positive windows jointly cover every non-zero mask pixel,
and negative windows are sampled deterministically from mask-free regions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageChops

from .deduplication import sha256_file
from .manifest import DatasetManifest
from .pipeline import load_samples_jsonl
from .schema import BoundingBox, DataSample, SchemaError

DEFAULT_DATASET_VERSION = "ksdd_tiled-1.0.0"
DEFAULT_PROMPT_VERSION = "ksdd_tile_audit_prompt-1.0.0"
DEFAULT_TILE_SIZES = (256, 384)
TEST_PROMPT_VARIANT = "tile_bilingual_direct"
TRAIN_PROMPT_VARIANTS = (
    TEST_PROMPT_VARIANT,
    "tile_bilingual_evidence_first",
    "tile_bilingual_quality_gate",
)

PROMPTS = {
    TEST_PROMPT_VARIANT: (
        "Inspect this commutator tile for a visible surface defect. Return exactly one JSON "
        "object and no Markdown. Use tile-local integer bbox [x1,y1,x2,y2]. A defect requires "
        "result=violation and surface_defect evidence; otherwise return result=compliant, "
        "objects=[], uncertain=false. / 检查该换向器切片是否存在可见表面缺陷。"
        "仅返回一个严格 JSON 对象，使用切片内整数 bbox [x1,y1,x2,y2]。有缺陷时 "
        "result=violation 并给出 surface_defect 证据；否则 result=compliant、objects=[]、"
        "uncertain=false。"
    ),
    "tile_bilingual_evidence_first": (
        "Examine the tile, localize visible anomaly evidence, then decide compliance. Output only "
        "strict JSON with tile-local integer xyxy coordinates. Positive means violation with a "
        "surface_defect bbox; negative means compliant with objects=[]; uncertain=false. / 先检查"
        "切片并定位可见异常证据，再判定合规性。仅输出严格 JSON，坐标为切片内整数 "
        "xyxy。正例为 violation 且含 surface_defect bbox；负例为 compliant 且 objects=[]；"
        "uncertain=false。"
    ),
    "tile_bilingual_quality_gate": (
        "Act as an industrial quality gate for this image tile. Return one JSON object only. "
        "Ground every visible defect with a tile-local integer surface_defect bbox; otherwise use "
        "compliant and an empty objects array. uncertain=false. / 作为该图像切片的工业质检"
        "门禁，仅返回一个 JSON 对象。每个可见缺陷必须使用切片内整数 surface_defect "
        "bbox 定位；否则使用 compliant 和空 objects 数组。uncertain=false。"
    ),
}


@dataclass(frozen=True)
class TileTransform:
    """Pure translation from a tile into its original image."""

    offset_x: int
    offset_y: int
    tile_width: int
    tile_height: int
    original_width: int
    original_height: int

    def __post_init__(self) -> None:
        values = (
            self.offset_x,
            self.offset_y,
            self.tile_width,
            self.tile_height,
            self.original_width,
            self.original_height,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise SchemaError("tile transform values must be integers")
        if self.offset_x < 0 or self.offset_y < 0:
            raise SchemaError("tile offsets must be non-negative")
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise SchemaError("tile dimensions must be positive")
        if self.offset_x + self.tile_width > self.original_width:
            raise SchemaError("tile exceeds original image width")
        if self.offset_y + self.tile_height > self.original_height:
            raise SchemaError("tile exceeds original image height")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "translation",
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "tile_xyxy_in_original": [
                self.offset_x,
                self.offset_y,
                self.offset_x + self.tile_width,
                self.offset_y + self.tile_height,
            ],
            "tile_to_original": {
                "x": "x_original=x_tile+offset_x",
                "y": "y_original=y_tile+offset_y",
            },
        }


@dataclass(frozen=True)
class TileRecord:
    tile_id: str
    source_sample_id: str
    entity_id: str
    split: str
    tile_size: int
    label: str
    negative_source: str | None
    image: str
    mask: str | None
    bbox_tile: BoundingBox | None
    bbox_original: BoundingBox | None
    transform: TileTransform
    source_image_sha256: str
    tile_image_sha256: str
    tile_mask_sha256: str | None

    @property
    def positive(self) -> bool:
        return self.label == "defect"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "source_sample_id": self.source_sample_id,
            "entity_id": self.entity_id,
            "split": self.split,
            "tile_size": self.tile_size,
            "label": self.label,
            "negative_source": self.negative_source,
            "image": self.image,
            "mask": self.mask,
            "bbox_tile": self.bbox_tile.as_list() if self.bbox_tile else None,
            "bbox_original": self.bbox_original.as_list() if self.bbox_original else None,
            "transform": self.transform.to_dict(),
            "source_image_sha256": self.source_image_sha256,
            "tile_image_sha256": self.tile_image_sha256,
            "tile_mask_sha256": self.tile_mask_sha256,
        }


@dataclass(frozen=True)
class KSDDTileExportResult:
    directory: Path
    dataset_version: str
    source_manifest_sha256: str
    tile_manifest_sha256: str
    tile_counts: Mapping[str, int]
    leakage_report: Mapping[str, Any]
    pilot_only: bool


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")


def _validate_tile_sizes(tile_sizes: Sequence[int]) -> tuple[int, ...]:
    values = tuple(tile_sizes)
    if not values or len(values) != len(set(values)):
        raise ValueError("tile sizes must be a non-empty unique sequence")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("tile sizes must be positive integers")
    return values


def _axis_origins(start: int, end: int, tile_size: int, limit: int) -> tuple[int, ...]:
    if tile_size > limit:
        raise SchemaError(f"tile size {tile_size} exceeds source dimension {limit}")
    if end - start <= tile_size:
        return (min(max((start + end - tile_size) // 2, 0), limit - tile_size),)
    origins = list(range(start, end - tile_size + 1, tile_size))
    origins.append(end - tile_size)
    return tuple(sorted({min(max(value, 0), limit - tile_size) for value in origins}))


def _positive_origins(mask: Image.Image, tile_size: int) -> tuple[tuple[int, int], ...]:
    binary = mask.convert("L").point(lambda value: 255 if value else 0)
    bbox = binary.getbbox()
    if bbox is None:
        return ()
    width, height = binary.size
    x_values = _axis_origins(bbox[0], bbox[2], tile_size, width)
    y_values = _axis_origins(bbox[1], bbox[3], tile_size, height)
    origins = [
        (x, y)
        for y in y_values
        for x in x_values
        if binary.crop((x, y, x + tile_size, y + tile_size)).getbbox() is not None
    ]
    covered = Image.new("L", binary.size, 0)
    for x, y in origins:
        covered.paste(255, (x, y, x + tile_size, y + tile_size))
    missing = ImageChops.subtract(binary, ImageChops.multiply(binary, covered))
    while missing.getbbox() is not None:
        point_box = missing.getbbox()
        assert point_box is not None
        x = min(max(point_box[0] - tile_size // 2, 0), width - tile_size)
        y = min(max(point_box[1] - tile_size // 2, 0), height - tile_size)
        origins.append((x, y))
        covered.paste(255, (x, y, x + tile_size, y + tile_size))
        missing = ImageChops.subtract(binary, ImageChops.multiply(binary, covered))
    return tuple(sorted(set(origins), key=lambda item: (item[1], item[0])))


def _deterministic_origin(
    sample_id: str,
    tile_size: int,
    width: int,
    height: int,
    mask: Image.Image | None,
) -> tuple[int, int]:
    if tile_size > width or tile_size > height:
        raise SchemaError(f"tile size {tile_size} exceeds {width}x{height} source {sample_id}")
    max_x, max_y = width - tile_size, height - tile_size
    for attempt in range(4096):
        digest = hashlib.sha256(
            f"{sample_id}:{tile_size}:negative:{attempt}".encode("utf-8")
        ).digest()
        x = int.from_bytes(digest[:8], "big") % (max_x + 1)
        y = int.from_bytes(digest[8:16], "big") % (max_y + 1)
        if mask is None or mask.crop((x, y, x + tile_size, y + tile_size)).getbbox() is None:
            return x, y
    step = max(1, tile_size // 4)
    for y in (*range(0, max_y + 1, step), max_y):
        for x in (*range(0, max_x + 1, step), max_x):
            if mask is None or mask.crop((x, y, x + tile_size, y + tile_size)).getbbox() is None:
                return x, y
    raise SchemaError(f"no mask-free {tile_size}px negative tile exists for {sample_id}")


def _map_local_bbox(box: BoundingBox, transform: TileTransform) -> BoundingBox:
    return BoundingBox(
        box.x1 + transform.offset_x,
        box.y1 + transform.offset_y,
        box.x2 + transform.offset_x,
        box.y2 + transform.offset_y,
    )


def _prompt_variant(tile_id: str, split: str) -> str:
    if split != "train":
        return TEST_PROMPT_VARIANT
    digest = hashlib.sha256(tile_id.encode("utf-8")).digest()
    return TRAIN_PROMPT_VARIANTS[int.from_bytes(digest[:2], "big") % len(TRAIN_PROMPT_VARIANTS)]


def _sft_record(
    record: TileRecord,
    image_reference: str,
    dataset_version: str,
    prompt_version: str,
) -> dict[str, Any]:
    split = "valid" if record.split == "validation" else record.split
    variant = _prompt_variant(record.tile_id, split)
    objects = (
        [{"label": "surface_defect", "bbox": record.bbox_tile.as_list()}]
        if record.bbox_tile
        else []
    )
    answer = {
        "result": "violation" if objects else "compliant",
        "objects": objects,
        "reason": (
            "A mask-confirmed defect is visible in this tile. / 该切片内存在掩码确认的可见缺陷。"
            if objects
            else "No defect pixels are present in this tile. / 该切片内不存在缺陷像素。"
        ),
        "uncertain": False,
    }
    return {
        "sample_id": record.tile_id,
        "source_sample_id": record.source_sample_id,
        "entity_id": record.entity_id,
        "split": split,
        "tile_size": record.tile_size,
        "dataset_version": dataset_version,
        "prompt_version": prompt_version,
        "prompt_variant": variant,
        "coordinate_space": "tile_local_xyxy",
        "images": [image_reference],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_reference},
                    {"type": "text", "text": PROMPTS[variant]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": _compact_json(answer)}],
            },
        ],
    }


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _coverage_ratio(mask: Image.Image, records: Sequence[TileRecord]) -> float:
    binary = mask.convert("L").point(lambda value: 255 if value else 0)
    total = sum(binary.histogram()[1:])
    if total == 0:
        return 1.0
    covered = Image.new("L", binary.size, 0)
    for record in records:
        transform = record.transform
        covered.paste(
            255,
            (
                transform.offset_x,
                transform.offset_y,
                transform.offset_x + transform.tile_width,
                transform.offset_y + transform.tile_height,
            ),
        )
    selected = ImageChops.multiply(binary, covered)
    return sum(selected.histogram()[1:]) / total


def _tile_record(
    *,
    staging: Path,
    sample: DataSample,
    split: str,
    tile_size: int,
    kind: str,
    index: int,
    source_image_sha256: str,
    image: Image.Image,
    mask: Image.Image,
    origin: tuple[int, int],
    negative_source: str | None,
) -> TileRecord:
    x, y = origin
    transform = TileTransform(x, y, tile_size, tile_size, image.width, image.height)
    tile_id = f"{sample.sample_id}_t{tile_size}_{kind}{index:02d}"
    image_relative = Path("tiles") / str(tile_size) / "images" / f"{tile_id}.png"
    mask_relative = Path("tiles") / str(tile_size) / "masks" / f"{tile_id}.png"
    tile_image = image.crop((x, y, x + tile_size, y + tile_size)).convert("RGB")
    tile_mask = mask.crop((x, y, x + tile_size, y + tile_size)).convert("L").point(
        lambda value: 255 if value else 0
    )
    local_box_value = tile_mask.getbbox()
    local_box = BoundingBox(*local_box_value) if local_box_value else None
    if kind == "pos" and local_box is None:
        raise SchemaError(f"positive tile contains no mask pixel: {tile_id}")
    if kind == "neg" and local_box is not None:
        raise SchemaError(f"negative tile intersects mask: {tile_id}")
    image_path = staging / image_relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    tile_image.save(image_path, format="PNG", optimize=False)
    mask_path: Path | None = None
    if local_box is not None:
        mask_path = staging / mask_relative
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        tile_mask.save(mask_path, format="PNG", optimize=False)
    return TileRecord(
        tile_id=tile_id,
        source_sample_id=sample.sample_id,
        entity_id=sample.entity_id,
        split=split,
        tile_size=tile_size,
        label="defect" if local_box else "good",
        negative_source=negative_source,
        image=image_relative.as_posix(),
        mask=mask_relative.as_posix() if mask_path else None,
        bbox_tile=local_box,
        bbox_original=_map_local_bbox(local_box, transform) if local_box else None,
        transform=transform,
        source_image_sha256=source_image_sha256,
        tile_image_sha256=sha256_file(image_path),
        tile_mask_sha256=sha256_file(mask_path) if mask_path else None,
    )


def _write_anomalib_export(staging: Path, records: Sequence[TileRecord]) -> None:
    for size in sorted({record.tile_size for record in records}):
        rows = []
        selected = [record for record in records if record.tile_size == size]
        for record in selected:
            image_source = staging / record.image
            image_relative = (
                Path("anomalib")
                / str(size)
                / record.split
                / record.label
                / f"{record.tile_id}.png"
            )
            _link_or_copy(image_source, staging / image_relative)
            mask_relative = ""
            if record.mask:
                mask_relative_path = (
                    Path("anomalib")
                    / str(size)
                    / record.split
                    / "mask"
                    / "defect"
                    / f"{record.tile_id}.png"
                )
                _link_or_copy(staging / record.mask, staging / mask_relative_path)
                mask_relative = mask_relative_path.as_posix()
            rows.append(
                {
                    "image_path": image_relative.as_posix(),
                    "split": "val" if record.split == "validation" else record.split,
                    "label": "anomalous" if record.positive else "normal",
                    "label_index": 1 if record.positive else 0,
                    "mask_path": mask_relative,
                    "tile_id": record.tile_id,
                    "source_sample_id": record.source_sample_id,
                    "entity_id": record.entity_id,
                    "transform_json": _compact_json(record.transform.to_dict()),
                }
            )
        csv_path = staging / "anomalib" / str(size) / "samples.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _write_text(
            staging / "anomalib" / str(size) / "samples.jsonl",
            "".join(_compact_json(row) + "\n" for row in rows),
        )


def _write_mlx_export(
    staging: Path,
    destination: Path,
    records: Sequence[TileRecord],
    dataset_version: str,
    prompt_version: str,
    reference_root: Path,
) -> None:
    for size in sorted({record.tile_size for record in records}):
        grouped: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
        for record in records:
            if record.tile_size != size:
                continue
            final_image = destination / record.image
            try:
                image_reference = final_image.relative_to(reference_root).as_posix()
            except ValueError:
                image_reference = str(final_image)
            sft = _sft_record(record, image_reference, dataset_version, prompt_version)
            grouped[str(sft["split"])].append(sft)
        for split, values in grouped.items():
            _write_text(
                staging / "mlx_vlm" / str(size) / "hf" / f"{split}.jsonl",
                "".join(_compact_json(value) + "\n" for value in values),
            )


def _leakage_report(
    records: Sequence[TileRecord], source_manifest_sha256: str
) -> dict[str, Any]:
    entities: dict[str, set[str]] = defaultdict(set)
    source_ids: dict[str, set[str]] = defaultdict(set)
    tile_hashes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        entities[record.split].add(record.entity_id)
        source_ids[record.split].add(record.source_sample_id)
        tile_hashes[record.split].add(record.tile_image_sha256)
    train_test_entity = sorted(entities["train"] & entities["test"])
    train_test_source = sorted(source_ids["train"] & source_ids["test"])
    train_test_tile_hash = sorted(tile_hashes["train"] & tile_hashes["test"])
    test_derived_training = sum(
        record.split == "train" and record.source_sample_id in source_ids["test"]
        for record in records
    )
    passes = not (
        train_test_entity or train_test_source or train_test_tile_hash or test_derived_training
    )
    return {
        "source_manifest_sha256": source_manifest_sha256,
        "split_inheritance": "each tile exactly inherits its frozen source split",
        "train_test_entity_overlap": train_test_entity,
        "train_test_source_sample_overlap": train_test_source,
        "train_test_tile_sha256_overlap": train_test_tile_hash,
        "test_derived_training_tile_count": test_derived_training,
        "source_assignments_preserved": True,
        "passes": passes,
    }


def _data_card(manifest: Mapping[str, Any]) -> str:
    stats = manifest["statistics"]
    sizes = ", ".join(str(value) for value in manifest["tile_sizes"])
    lines = [
        "# KSDD Tiled Specialist V1 data card",
        "",
        f"- Dataset version: `{manifest['dataset_version']}`",
        f"- Source dataset: `{manifest['source_dataset_version']}`",
        f"- Tile sizes: {sizes} pixels",
        f"- Total tiles: {stats['total_tiles']}",
        f"- Positive/negative: {stats['positive_tiles']} / {stats['negative_tiles']}",
        "- License: CC BY-NC-SA 4.0; research/non-commercial use only.",
        "- Status: **pilot_only**. No model was trained or evaluated by this export.",
        "",
        "## Construction",
        "",
        "Each tile inherits the frozen physical-entity split. Positive windows jointly cover 100% "
        "of the source mask pixels; their local boxes are re-derived from cropped masks. One "
        "deterministic mask-free negative is drawn from every source image for each size. On defect "
        "images this is a same-image non-defect region; on normal images it is a normal-image crop.",
        "",
        "## Intended use and limits",
        "",
        "MLX-VLM JSONL supports tile-level visual-compliance SFT. The Anomalib export includes "
        "split folders plus a sample manifest with the standard image_path, split, label_index and "
        "mask_path fields. One-class Anomalib methods should train on train/good; supervised "
        "specialists may consume positive rows from samples.csv. Tile metrics are pilot evidence, "
        "not formal KPI results. KSDD has narrow controlled imagery and only nine positive test "
        "source images, so generalization and confidence estimates are limited.",
        "",
    ]
    return "\n".join(lines)


def export_ksdd_tiles(
    source_root: str | Path,
    output_root: str | Path,
    *,
    expected_source_manifest_sha256: str,
    created_at: str,
    record_schema_path: str | Path,
    sft_record_schema_path: str | Path,
    sft_answer_schema_path: str | Path,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    tile_sizes: Sequence[int] = DEFAULT_TILE_SIZES,
    reference_root: str | Path | None = None,
) -> KSDDTileExportResult:
    """Export MLX-VLM and Anomalib tiled derivatives without changing source splits."""

    _validate_timestamp(created_at)
    sizes = _validate_tile_sizes(tile_sizes)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*-\d+\.\d+\.\d+", dataset_version):
        raise ValueError("dataset_version must match name-major.minor.patch")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*-\d+\.\d+\.\d+", prompt_version):
        raise ValueError("prompt_version must match name-major.minor.patch")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_manifest_sha256):
        raise ValueError("expected source manifest hash must be lowercase SHA-256")
    source = Path(source_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite tiled dataset: {destination}")
    manifest_path = source / "manifest.json"
    actual_source_hash = sha256_file(manifest_path)
    if actual_source_hash != expected_source_manifest_sha256:
        raise SchemaError("source manifest hash mismatch; refusing unfrozen input")
    source_manifest = DatasetManifest.load(manifest_path)
    if not source_manifest.frozen_test:
        raise SchemaError("source test split must be frozen")
    samples = load_samples_jsonl(source / "samples.jsonl")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    entry_by_id = {entry.sample_id: entry for entry in source_manifest.entries}
    if set(sample_by_id) != set(entry_by_id):
        raise SchemaError("source samples and manifest entries differ")

    schema_paths = tuple(
        Path(value).resolve()
        for value in (record_schema_path, sft_record_schema_path, sft_answer_schema_path)
    )
    if not all(path.is_file() for path in schema_paths):
        raise FileNotFoundError("tile and SFT schema files are required")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        records: list[TileRecord] = []
        coverage: dict[str, float] = {}
        bbox_reconstruction: dict[str, bool] = {}
        for entry in sorted(source_manifest.entries, key=lambda item: item.sample_id):
            sample = sample_by_id[entry.sample_id]
            image_path = source / entry.image
            if sha256_file(image_path) != entry.sha256:
                raise SchemaError(f"source image hash changed: {entry.sample_id}")
            mask_relative = sample.metadata.get("mask")
            if not isinstance(mask_relative, str):
                raise SchemaError(f"source mask metadata missing: {entry.sample_id}")
            mask_path = source / mask_relative
            with Image.open(image_path) as opened_image, Image.open(mask_path) as opened_mask:
                image = opened_image.convert("RGB")
                mask = opened_mask.convert("L").point(lambda value: 255 if value else 0)
            if image.size != mask.size:
                raise SchemaError(f"source image/mask mismatch: {entry.sample_id}")
            for size in sizes:
                positive_records = []
                for index, origin in enumerate(_positive_origins(mask, size)):
                    record = _tile_record(
                        staging=staging,
                        sample=sample,
                        split=entry.split,
                        tile_size=size,
                        kind="pos",
                        index=index,
                        source_image_sha256=entry.sha256,
                        image=image,
                        mask=mask,
                        origin=origin,
                        negative_source=None,
                    )
                    records.append(record)
                    positive_records.append(record)
                if positive_records:
                    ratio = _coverage_ratio(mask, positive_records)
                    coverage[f"{entry.sample_id}:{size}"] = ratio
                    if ratio != 1.0:
                        raise SchemaError(
                            f"positive tiles do not fully cover source mask {entry.sample_id}: {ratio}"
                        )
                    mapped_boxes = [record.bbox_original for record in positive_records]
                    if any(box is None for box in mapped_boxes):
                        raise SchemaError(f"positive tile lacks mapped bbox: {entry.sample_id}")
                    reconstructed = BoundingBox(
                        min(box.x1 for box in mapped_boxes if box is not None),
                        min(box.y1 for box in mapped_boxes if box is not None),
                        max(box.x2 for box in mapped_boxes if box is not None),
                        max(box.y2 for box in mapped_boxes if box is not None),
                    )
                    source_boxes = [item.bbox for item in sample.response.objects]
                    if len(source_boxes) != 1 or reconstructed != source_boxes[0]:
                        raise SchemaError(
                            f"mapped tile bboxes do not reconstruct source bbox: {entry.sample_id}"
                        )
                    bbox_reconstruction[f"{entry.sample_id}:{size}"] = True
                negative_origin = _deterministic_origin(
                    entry.sample_id,
                    size,
                    image.width,
                    image.height,
                    mask if mask.getbbox() is not None else None,
                )
                records.append(
                    _tile_record(
                        staging=staging,
                        sample=sample,
                        split=entry.split,
                        tile_size=size,
                        kind="neg",
                        index=0,
                        source_image_sha256=entry.sha256,
                        image=image,
                        mask=mask,
                        origin=negative_origin,
                        negative_source=(
                            "same_image_nondefect_region"
                            if mask.getbbox() is not None
                            else "normal_image"
                        ),
                    )
                )

        leakage = _leakage_report(records, actual_source_hash)
        if not leakage["passes"]:
            raise SchemaError(f"tiled split leakage gate failed: {leakage}")

        _write_text(
            staging / "tile_records.jsonl",
            "".join(_compact_json(record.to_dict()) + "\n" for record in records),
        )
        for schema in schema_paths:
            target_name = (
                "ksdd_tile_record.schema.json"
                if schema == schema_paths[0]
                else schema.name
            )
            target = staging / "schema" / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(schema, target)

        final_reference_root = (
            Path(reference_root).resolve() if reference_root is not None else Path.cwd().resolve()
        )
        _write_mlx_export(
            staging,
            destination,
            records,
            dataset_version,
            prompt_version,
            final_reference_root,
        )
        _write_anomalib_export(staging, records)
        _write_json(staging / "leakage_report.json", leakage)

        split_counts = Counter(record.split for record in records)
        size_counts = Counter(str(record.tile_size) for record in records)
        positives = sum(record.positive for record in records)
        negative_sources = Counter(
            record.negative_source for record in records if record.negative_source is not None
        )
        source_test_positive_count = sum(
            entry.split == "test" and bool(sample_by_id[entry.sample_id].response.objects)
            for entry in source_manifest.entries
        )
        counts_by_size_and_split = {
            str(size): {
                split: {
                    "total": len(selected := [
                        record
                        for record in records
                        if record.tile_size == size and record.split == split
                    ]),
                    "positive": sum(record.positive for record in selected),
                    "negative": sum(not record.positive for record in selected),
                    "positive_ratio": (
                        sum(record.positive for record in selected) / len(selected)
                        if selected
                        else 0.0
                    ),
                }
                for split in ("train", "validation", "test")
            }
            for size in sizes
        }
        statistics = {
            "total_tiles": len(records),
            "tile_counts_by_size": dict(sorted(size_counts.items())),
            "tile_counts_by_split": dict(sorted(split_counts.items())),
            "positive_tiles": positives,
            "negative_tiles": len(records) - positives,
            "positive_ratio": positives / len(records) if records else 0.0,
            "negative_ratio": (len(records) - positives) / len(records) if records else 0.0,
            "negative_source_counts": dict(sorted(negative_sources.items())),
            "counts_by_size_and_split": counts_by_size_and_split,
            "positive_source_size_coverage_min": min(coverage.values(), default=1.0),
            "positive_source_bbox_reconstruction_rate": (
                sum(bbox_reconstruction.values()) / len(bbox_reconstruction)
                if bbox_reconstruction
                else 1.0
            ),
            "source_test_samples": sum(entry.split == "test" for entry in source_manifest.entries),
            "source_test_positive_samples": source_test_positive_count,
            "formal_kpi_eligible": False,
            "evaluation_status": "pilot_only",
            "formal_kpi_ineligibility_reason": (
                f"frozen source test split has only {source_test_positive_count} positive images"
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": "1.0.0",
            "dataset_version": dataset_version,
            "source_dataset_version": source_manifest.dataset_version,
            "source_manifest_sha256": actual_source_hash,
            "created_at": created_at,
            "tile_sizes": list(sizes),
            "prompt_version": prompt_version,
            "split_policy": {
                "inherit_frozen_source_split": True,
                "entity_isolated": True,
                "test_derived_training_tiles": 0,
                "test_prompt_fixed": True,
            },
            "sampling_policy": {
                "positive": "deterministic windows with 100% source-mask pixel coverage",
                "positive_bbox": "tight bbox re-derived from each cropped binary mask",
                "negative_defect_image": "one deterministic mask-free same-image region per size",
                "negative_normal_image": "one deterministic crop per normal source image per size",
                "transform": "lossless crop; tile-local to original is pure xy translation",
            },
            "exports": {
                "mlx_vlm": "mlx_vlm/{tile_size}/hf/{train,valid,test}.jsonl",
                "anomalib_folders": (
                    "anomalib/{tile_size}/{train,validation,test}/{good,defect,mask/defect}"
                ),
                "anomalib_manifest": "anomalib/{tile_size}/samples.csv",
            },
            "statistics": statistics,
            "license": source_manifest.licenses["cc-by-nc-sa-4.0"].to_dict(),
            "leakage_report": "leakage_report.json",
            "pilot_only": True,
            "model_run": False,
            "model_metrics": None,
        }
        _write_text(staging / "DATA_CARD.md", _data_card(manifest))

        file_hashes = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest["file_sha256"] = file_hashes
        _write_json(staging / "tile_manifest.json", manifest)
        manifest_hash = sha256_file(staging / "tile_manifest.json")
        _write_text(staging / "tile_manifest.sha256", f"{manifest_hash}  tile_manifest.json\n")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return KSDDTileExportResult(
        directory=destination,
        dataset_version=dataset_version,
        source_manifest_sha256=actual_source_hash,
        tile_manifest_sha256=manifest_hash,
        tile_counts=dict(sorted(size_counts.items())),
        leakage_report=leakage,
        pilot_only=True,
    )


def load_tile_records(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load the canonical JSONL tile manifest for downstream adapters."""

    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"invalid tile JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, Mapping):
                raise SchemaError(f"tile JSONL line {line_number} must be an object")
            rows.append(value)
    return tuple(rows)
