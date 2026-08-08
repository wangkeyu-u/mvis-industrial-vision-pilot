"""Image, annotation, and dataset-level validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from .schema import DataSample

SUPPORTED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
FORMAT_EXTENSIONS = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "WEBP": {".webp"}}


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    format: str
    mime_type: str
    size_bytes: int


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    sample_id: str | None = None


@dataclass
class DatasetValidationReport:
    checked_samples: int = 0
    image_metadata: dict[str, ImageMetadata] = field(default_factory=dict)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def _has_matching_signature(path: Path, image_format: str) -> bool:
    with path.open("rb") as stream:
        header = stream.read(16)
    if image_format == "JPEG":
        return header.startswith(b"\xff\xd8\xff")
    if image_format == "PNG":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if image_format == "WEBP":
        return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    return False


def validate_image(
    path: str | Path,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    max_pixels: int = 25_000_000,
) -> ImageMetadata:
    image_path = Path(path)
    if not image_path.is_file():
        raise ValueError(f"image does not exist: {image_path}")
    size_bytes = image_path.stat().st_size
    if size_bytes <= 0 or size_bytes > max_bytes:
        raise ValueError(f"image size must be between 1 and {max_bytes} bytes")
    try:
        with Image.open(image_path) as image:
            image_format = image.format
            if image_format not in SUPPORTED_FORMATS:
                raise ValueError(f"unsupported decoded image format: {image_format}")
            if not _has_matching_signature(image_path, image_format):
                raise ValueError("file signature does not match decoded image format")
            if image_path.suffix.lower() not in FORMAT_EXTENSIONS[image_format]:
                raise ValueError("file extension does not match decoded image format")
            image.verify()
        with Image.open(image_path) as image:
            oriented_image = ImageOps.exif_transpose(image)
            width, height = oriented_image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError(f"decoded image exceeds the {max_pixels} pixel limit")
            oriented_image.load()
    except (UnidentifiedImageError, OSError, RuntimeError, Image.DecompressionBombError) as exc:
        raise ValueError(f"image cannot be decoded: {image_path}") from exc
    return ImageMetadata(width, height, image_format, SUPPORTED_FORMATS[image_format], size_bytes)


def _resolve_inside(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("image path escapes dataset root")
    return candidate


def validate_dataset(
    samples: Sequence[DataSample],
    dataset_root: str | Path,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    max_pixels: int = 25_000_000,
) -> DatasetValidationReport:
    report = DatasetValidationReport(checked_samples=len(samples))
    seen_ids: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen_ids:
            report.errors.append(
                ValidationIssue("DUPLICATE_SAMPLE_ID", "sample_id is not unique", sample.sample_id)
            )
            continue
        seen_ids.add(sample.sample_id)
        try:
            path = _resolve_inside(Path(dataset_root), sample.image)
            metadata = validate_image(path, max_bytes=max_bytes, max_pixels=max_pixels)
            report.image_metadata[sample.sample_id] = metadata
        except ValueError as exc:
            report.errors.append(ValidationIssue("INVALID_IMAGE", str(exc), sample.sample_id))
            continue
        for annotation in sample.response.objects:
            if not annotation.bbox.within(metadata.width, metadata.height):
                report.errors.append(
                    ValidationIssue(
                        "BBOX_OUT_OF_BOUNDS",
                        f"bbox {annotation.bbox.as_list()} exceeds image {metadata.width}x{metadata.height}",
                        sample.sample_id,
                    )
                )
        if sample.is_hard_negative and sample.response.objects:
            report.errors.append(
                ValidationIssue(
                    "HARD_NEGATIVE_HAS_OBJECTS",
                    "hard-negative ground truth must not contain positive objects",
                    sample.sample_id,
                )
            )
    return report
