"""Versioned dataset manifest with license and integrity metadata."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .deduplication import perceptual_hash, sha256_file
from .schema import DataSample, LicenseInfo, SchemaError
from .validation import validate_dataset

VERSION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*-\d+\.\d+\.\d+$")
VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    entity_id: str
    image: str
    split: str
    sha256: str
    perceptual_hash: str
    source: str
    license_id: str

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise SchemaError(f"invalid split: {self.split!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise SchemaError("sha256 must contain 64 lowercase hexadecimal characters")
        if not re.fullmatch(r"[0-9a-f]+", self.perceptual_hash):
            raise SchemaError("perceptual_hash must be lowercase hexadecimal")
        for field_name in ("sample_id", "entity_id", "image", "source", "license_id"):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name).strip():
                raise SchemaError(f"manifest entry {field_name} must be non-empty")
        image_path = PurePosixPath(self.image)
        if image_path.is_absolute() or ".." in image_path.parts:
            raise SchemaError("manifest image must be a safe relative path")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManifestEntry":
        return cls(**{name: value.get(name) for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "entity_id": self.entity_id,
            "image": self.image,
            "split": self.split,
            "sha256": self.sha256,
            "perceptual_hash": self.perceptual_hash,
            "source": self.source,
            "license_id": self.license_id,
        }


@dataclass(frozen=True)
class DatasetManifest:
    dataset_version: str
    schema_version: str
    created_at: str
    entries: tuple[ManifestEntry, ...]
    licenses: Mapping[str, LicenseInfo]
    split_strategy: Mapping[str, Any]
    frozen_test: bool
    parent_version: str | None = None
    change_summary: str | None = None
    statistics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not VERSION_PATTERN.fullmatch(self.dataset_version):
            raise SchemaError("dataset_version must match domain-major.minor.patch")
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.schema_version):
            raise SchemaError("schema_version must use semantic version format")
        try:
            parsed_created_at = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise SchemaError("created_at must be an ISO-8601 timestamp") from exc
        if parsed_created_at.tzinfo is None:
            raise SchemaError("created_at must include a timezone")
        if not isinstance(self.frozen_test, bool):
            raise SchemaError("frozen_test must be a boolean")
        if self.parent_version is not None and not VERSION_PATTERN.fullmatch(self.parent_version):
            raise SchemaError("parent_version must match domain-major.minor.patch")
        if not isinstance(self.split_strategy, Mapping) or not isinstance(self.statistics, Mapping):
            raise SchemaError("split_strategy and statistics must be objects")
        ids = [entry.sample_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise SchemaError("manifest sample_id values must be unique")
        missing = sorted({entry.license_id for entry in self.entries} - set(self.licenses))
        if missing:
            raise SchemaError(f"manifest entries reference unknown licenses: {missing}")
        if set(entry.split for entry in self.entries) - VALID_SPLITS:
            raise SchemaError("manifest contains invalid splits")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifest":
        raw_licenses = value.get("licenses", {})
        if not isinstance(raw_licenses, Mapping):
            raise SchemaError("licenses must be an object keyed by identifier")
        licenses = {key: LicenseInfo.from_dict(item) for key, item in raw_licenses.items()}
        if any(key != license_info.identifier for key, license_info in licenses.items()):
            raise SchemaError("license registry key must equal license.identifier")
        return cls(
            dataset_version=value.get("dataset_version"),
            schema_version=value.get("schema_version"),
            created_at=value.get("created_at"),
            entries=tuple(ManifestEntry.from_dict(item) for item in value.get("entries", [])),
            licenses=licenses,
            split_strategy=value.get("split_strategy", {}),
            frozen_test=value.get("frozen_test", False),
            parent_version=value.get("parent_version"),
            change_summary=value.get("change_summary"),
            statistics=value.get("statistics", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "parent_version": self.parent_version,
            "change_summary": self.change_summary,
            "frozen_test": self.frozen_test,
            "split_strategy": dict(self.split_strategy),
            "statistics": dict(self.statistics),
            "licenses": {key: value.to_dict() for key, value in sorted(self.licenses.items())},
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def write_atomic(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def split_statistics(entries: Sequence[ManifestEntry]) -> dict[str, int]:
    return {split: sum(entry.split == split for entry in entries) for split in sorted(VALID_SPLITS)}


def build_manifest(
    samples: Sequence[DataSample],
    dataset_root: str | Path,
    assignments: Mapping[str, str],
    *,
    dataset_version: str,
    created_at: str,
    schema_version: str = "1.0.0",
    frozen_test: bool = False,
    split_strategy: Mapping[str, Any] | None = None,
    parent_version: str | None = None,
    change_summary: str | None = None,
) -> DatasetManifest:
    """Validate samples and build a complete, checksummed release manifest."""

    report = validate_dataset(samples, dataset_root)
    if not report.valid:
        summary = "; ".join(
            f"{issue.sample_id or '<dataset>'}:{issue.code}" for issue in report.errors[:10]
        )
        raise SchemaError(f"dataset validation failed: {summary}")
    sample_ids = {sample.sample_id for sample in samples}
    if set(assignments) != sample_ids:
        missing = sorted(sample_ids - set(assignments))
        extra = sorted(set(assignments) - sample_ids)
        raise SchemaError(f"split assignments must exactly cover samples; missing={missing}, extra={extra}")

    root = Path(dataset_root).resolve()
    licenses: dict[str, LicenseInfo] = {}
    entries = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        existing_license = licenses.get(sample.license.identifier)
        if existing_license is not None and existing_license != sample.license:
            raise SchemaError(f"conflicting definitions for license {sample.license.identifier!r}")
        licenses[sample.license.identifier] = sample.license
        image_path = (root / sample.image).resolve()
        entries.append(
            ManifestEntry(
                sample_id=sample.sample_id,
                entity_id=sample.entity_id,
                image=sample.image,
                split=assignments[sample.sample_id],
                sha256=sha256_file(image_path),
                perceptual_hash=perceptual_hash(image_path),
                source=sample.source,
                license_id=sample.license.identifier,
            )
        )
    statistics: dict[str, Any] = split_statistics(entries)
    statistics.update(
        {
            "total": len(entries),
            "hard_negative": sum(sample.is_hard_negative for sample in samples),
            "license_count": len(licenses),
        }
    )
    return DatasetManifest(
        dataset_version=dataset_version,
        schema_version=schema_version,
        created_at=created_at,
        entries=tuple(entries),
        licenses=licenses,
        split_strategy=dict(split_strategy or {}),
        frozen_test=frozen_test,
        parent_version=parent_version,
        change_summary=change_summary,
        statistics=statistics,
    )
