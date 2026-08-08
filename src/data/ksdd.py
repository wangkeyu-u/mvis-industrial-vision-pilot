"""KolektorSDD adapter for the canonical visual-compliance dataset contract."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from PIL import Image, ImageChops

from .data_card import generate_data_card
from .deduplication import DuplicateGroup, find_near_duplicate_groups, perceptual_hash, sha256_file
from .leakage import LeakageReport, detect_split_leakage
from .license_policy import LicensePolicy, require_allowed_license
from .manifest import DatasetManifest, build_manifest
from .schema import (
    BoundingBox,
    DataSample,
    LicenseInfo,
    ObjectAnnotation,
    SampleResponse,
    SchemaError,
)
from .splitting import SplitRatios, entity_isolated_split
from .statistics import compute_dataset_statistics
from .validation import DatasetValidationReport, validate_dataset

KSDD_SOURCE_PAGE = "https://www.vicos.si/resources/kolektorsdd/"
KSDD_ARCHIVE_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD.zip"
KSDD_V0_ARCHIVE_SHA256 = "65dc621693418585de9c4467d1340ea7958a6181816f0dc2883a1e8b61f9d4dc"
KSDD_V0_ARCHIVE_BYTES = 101_831_129
KSDD_LICENSE = LicenseInfo(
    identifier="cc-by-nc-sa-4.0",
    name="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
    url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    attribution=(
        "Kolektor Group d.o.o.; Domen Tabernik, Samo Sela, Jure Skvarc, "
        "and Danijel Skocaj, Visual Cognitive Systems Laboratory, University of Ljubljana"
    ),
    redistributable=True,
    commercial_use=False,
    derivative_work=True,
    notes="Non-commercial use only; adapted annotations must be shared alike.",
)
KSDD_INSTRUCTION = (
    "Inspect this electrical-commutator surface for a visible production defect and localize "
    "the defect if present. / 检查该电机换向器表面是否存在可见生产缺陷，并在存在时定位缺陷。"
)


@dataclass(frozen=True)
class KSDDSourceRecord:
    source_page: str
    archive_url: str
    archive_sha256: str
    archive_bytes: int
    downloaded_at: str

    def __post_init__(self) -> None:
        if not self.source_page.startswith("https://") or not self.archive_url.startswith("https://"):
            raise ValueError("KSDD source URLs must use HTTPS")
        if len(self.archive_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.archive_sha256
        ):
            raise ValueError("archive_sha256 must be 64 lowercase hexadecimal characters")
        if self.archive_bytes <= 0 or self.archive_bytes > 2 * 1024**3:
            raise ValueError("archive must be between 1 byte and 2 GiB")
        parsed = datetime.fromisoformat(self.downloaded_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("downloaded_at must include a timezone")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": "Kolektor Surface-Defect Dataset (KSDD)",
            "publisher": "ViCoS Lab, University of Ljubljana / Kolektor Group d.o.o.",
            "source_page": self.source_page,
            "archive_url": self.archive_url,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "downloaded_at": self.downloaded_at,
            "license": KSDD_LICENSE.to_dict(),
        }


@dataclass(frozen=True)
class KSDDPreparationResult:
    samples: tuple[DataSample, ...]
    assignments: Mapping[str, str]
    duplicate_groups: tuple[DuplicateGroup, ...]
    exact_duplicate_removed_ids: tuple[str, ...]
    validation: DatasetValidationReport
    leakage: LeakageReport
    manifest: DatasetManifest
    statistics: Mapping[str, Any]
    probe_sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class _SourceItem:
    sample_id: str
    entity_id: str
    image_path: Path
    mask_path: Path
    defect_bbox: BoundingBox | None
    source_sample_ids: tuple[str, ...]
    mask_paths: tuple[Path, ...]
    exact_duplicate_annotation_conflict: bool = False


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _jsonl(samples: Sequence[DataSample]) -> str:
    return "".join(
        json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for sample in samples
    )


def _evaluation_ground_truth_jsonl(
    samples: Sequence[DataSample],
    assignments: Mapping[str, str],
    validation: DatasetValidationReport,
) -> str:
    records = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        if assignments[sample.sample_id] != "test":
            continue
        image = validation.image_metadata[sample.sample_id]
        records.append(
            {
                "sample_id": sample.sample_id,
                "response": sample.response.to_dict(),
                "image_width": image.width,
                "image_height": image.height,
                "difficulty": list(sample.difficulty),
                "source": sample.source,
                "slices": list(sample.metadata.get("slices", [])),
            }
        )
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )


def mask_bounding_box(mask_path: str | Path) -> BoundingBox | None:
    """Return the tight union box of non-zero mask pixels, or ``None`` for a good item."""

    with Image.open(mask_path) as mask:
        box = mask.convert("L").getbbox()
    return BoundingBox(*box) if box is not None else None


def discover_ksdd(source_root: str | Path) -> tuple[_SourceItem, ...]:
    """Discover and structurally validate the original fine-annotation KSDD release."""

    root = Path(source_root).resolve()
    images = sorted(root.glob("kos*/Part*.jpg"))
    if not images:
        raise SchemaError(f"no KSDD images found below {root}")
    items = []
    for image_path in images:
        if not image_path.parent.name.startswith("kos"):
            raise SchemaError(f"unexpected KSDD entity directory: {image_path.parent.name}")
        mask_path = image_path.with_name(f"{image_path.stem}_label.bmp")
        if not mask_path.is_file():
            raise SchemaError(f"missing KSDD mask for {image_path}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise SchemaError(
                    f"image/mask size mismatch for {image_path}: {image.size} != {mask.size}"
                )
        entity_id = f"ksdd_{image_path.parent.name.lower()}"
        sample_id = f"{entity_id}_{image_path.stem.lower()}"
        items.append(
            _SourceItem(
                sample_id=sample_id,
                entity_id=entity_id,
                image_path=image_path,
                mask_path=mask_path,
                defect_bbox=mask_bounding_box(mask_path),
                source_sample_ids=(sample_id,),
                mask_paths=(mask_path,),
            )
        )
    return tuple(items)


def _sample(item: _SourceItem, image: str, mask: str, *, split: str | None = None) -> DataSample:
    defect = item.defect_bbox is not None
    difficulty = ["industrial_surface", "grayscale", "fine_defect"]
    slices = ["domain:industrial", "object:commutator", "capture:controlled"]
    if defect:
        difficulty.extend(["positive", "mask_derived_bbox"])
        slices.extend(["result:defect", "annotation:pixel_mask"])
    else:
        difficulty.append("negative")
        slices.append("result:good")
        if split == "test":
            difficulty.append("hard_negative")
            slices.append("negative:unseen_physical_item")
    response = SampleResponse(
        result="violation" if defect else "compliant",
        objects=(ObjectAnnotation("surface_defect", item.defect_bbox),) if defect else (),
        reason=(
            "The official pixel mask marks a visible surface defect."
            if defect
            else "The official mask contains no defect pixels."
        ),
        uncertain=False,
    )
    return DataSample(
        sample_id=item.sample_id,
        entity_id=item.entity_id,
        image=image,
        instruction=KSDD_INSTRUCTION,
        response=response,
        difficulty=tuple(difficulty),
        source="vicos_ksdd_fine_annotations",
        license=KSDD_LICENSE,
        metadata={
            "source_page": KSDD_SOURCE_PAGE,
            "source_entity_directory": item.image_path.parent.name,
            "source_image": item.image_path.name,
            "mask": mask,
            "split": split,
            "slices": slices,
            "annotation_transform": "tight bounding box around all non-zero mask pixels",
            "source_sample_ids": list(item.source_sample_ids),
            "exact_duplicate_annotation_conflict": item.exact_duplicate_annotation_conflict,
            "exact_duplicate_annotation_policy": (
                "pixelwise union of official masks"
                if item.exact_duplicate_annotation_conflict
                else "not_applicable"
            ),
        },
    )


def _remove_exact_duplicates(items: Sequence[_SourceItem]) -> tuple[tuple[_SourceItem, ...], tuple[str, ...]]:
    grouped: dict[str, list[_SourceItem]] = {}
    for item in sorted(items, key=lambda value: value.sample_id):
        grouped.setdefault(sha256_file(item.image_path), []).append(item)
    kept = []
    removed = []
    for members in grouped.values():
        canonical = members[0]
        boxes = [member.defect_bbox for member in members if member.defect_bbox is not None]
        merged_box = (
            BoundingBox(
                min(box.x1 for box in boxes),
                min(box.y1 for box in boxes),
                max(box.x2 for box in boxes),
                max(box.y2 for box in boxes),
            )
            if boxes
            else None
        )
        annotation_boxes = {
            tuple(member.defect_bbox.as_list()) if member.defect_bbox is not None else None
            for member in members
        }
        conflict = len(annotation_boxes) > 1
        kept.append(
            replace(
                canonical,
                defect_bbox=merged_box,
                source_sample_ids=tuple(member.sample_id for member in members),
                mask_paths=tuple(member.mask_path for member in members),
                exact_duplicate_annotation_conflict=conflict,
            )
        )
        removed.extend(member.sample_id for member in members[1:])
    return tuple(kept), tuple(removed)


def _copy_items(items: Sequence[_SourceItem], output_root: Path) -> dict[str, tuple[str, str]]:
    paths = {}
    for item in items:
        image_relative = PurePosixPath("images", item.image_path.parent.name, item.image_path.name)
        mask_relative = PurePosixPath("masks", item.mask_path.parent.name, item.mask_path.name)
        image_destination = output_root / image_relative
        mask_destination = output_root / mask_relative
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.image_path, image_destination)
        if len(item.mask_paths) == 1:
            shutil.copy2(item.mask_path, mask_destination)
        else:
            with Image.open(item.mask_paths[0]) as first_mask:
                merged_mask = first_mask.convert("L")
            for source_mask_path in item.mask_paths[1:]:
                with Image.open(source_mask_path) as source_mask:
                    merged_mask = ImageChops.lighter(merged_mask, source_mask.convert("L"))
            merged_mask.save(mask_destination)
        paths[item.sample_id] = (image_relative.as_posix(), mask_relative.as_posix())
    return paths


def _select_probes(samples: Sequence[DataSample], assignments: Mapping[str, str]) -> tuple[str, ...]:
    test_samples = sorted(
        (sample for sample in samples if assignments[sample.sample_id] == "test"),
        key=lambda sample: sample.sample_id,
    )
    positives = [sample.sample_id for sample in test_samples if sample.response.objects]
    negatives = [sample.sample_id for sample in test_samples if not sample.response.objects]
    selected = positives[:3] + negatives[:2]
    if len(selected) < 5:
        selected.extend(
            sample.sample_id
            for sample in sorted(samples, key=lambda item: item.sample_id)
            if sample.sample_id not in selected
        )
    if len(selected) < 5:
        raise SchemaError("KSDD conversion requires at least five test probe samples")
    return tuple(selected[:5])


def prepare_ksdd_dataset(
    source_root: str | Path,
    output_root: str | Path,
    source_record: KSDDSourceRecord,
    policy: LicensePolicy,
    *,
    dataset_version: str = "ksdd-0.1.0",
    created_at: str,
    seed: int = 42,
    ratios: SplitRatios = SplitRatios(),
    max_hash_distance: int = 5,
    minimum_formal_kpi_test_samples: int = 300,
) -> KSDDPreparationResult:
    """Materialize, split, validate, and freeze a canonical KSDD V0 package."""

    if minimum_formal_kpi_test_samples <= 0:
        raise ValueError("minimum_formal_kpi_test_samples must be positive")
    require_allowed_license(policy, KSDD_LICENSE)
    destination = Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty frozen dataset output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    discovered = discover_ksdd(source_root)
    unique_items, removed_ids = _remove_exact_duplicates(discovered)
    materialized_paths = _copy_items(unique_items, destination)
    provisional = tuple(
        _sample(item, *materialized_paths[item.sample_id]) for item in unique_items
    )
    hashes = {
        sample.sample_id: perceptual_hash(destination / sample.image) for sample in provisional
    }
    duplicate_groups = find_near_duplicate_groups(hashes, max_distance=max_hash_distance)
    assignments = entity_isolated_split(provisional, duplicate_groups, ratios=ratios, seed=seed)
    samples = tuple(
        _sample(
            item,
            *materialized_paths[item.sample_id],
            split=assignments[item.sample_id],
        )
        for item in unique_items
    )
    validation = validate_dataset(samples, destination)
    if not validation.valid:
        raise SchemaError(f"KSDD validation failed: {[issue.code for issue in validation.errors[:10]]}")
    leakage = detect_split_leakage(samples, assignments, duplicate_groups)
    if not leakage.passes(0.0):
        raise SchemaError(f"KSDD split leakage gate failed: {leakage.to_dict()}")

    manifest = build_manifest(
        samples,
        destination,
        assignments,
        dataset_version=dataset_version,
        created_at=created_at,
        frozen_test=True,
        split_strategy={
            "algorithm": "physical_entity_and_perceptual_hash_component_greedy",
            "entity_key": "official kosNN physical item directory",
            "seed": seed,
            "ratios": ratios.as_dict(),
            "perceptual_hash": "dhash-64",
            "max_hamming_distance": max_hash_distance,
            "exact_duplicates": "sha256_remove_after_annotation_consistency_check",
            "test_policy": "frozen at V0; no retuning against test",
        },
        change_summary="Initial public-data V0 pilot from the official fine-annotation KSDD release.",
    )
    statistics = compute_dataset_statistics(samples, assignments)
    split_results = Counter(
        (assignments[sample.sample_id], sample.response.result) for sample in samples
    )
    test_count = int(statistics["split_counts"].get("test", 0))
    test_hard_negative_count = sum(
        sample.is_hard_negative
        for sample in samples
        if assignments[sample.sample_id] == "test"
    )
    formal_kpi_eligible = test_count >= minimum_formal_kpi_test_samples
    extra_statistics = {
        "source_discovered": len(discovered),
        "exact_duplicate_removed": len(removed_ids),
        "exact_duplicate_annotation_conflicts_merged": sum(
            item.exact_duplicate_annotation_conflict for item in unique_items
        ),
        "near_duplicate_groups": len(duplicate_groups),
        "near_duplicate_samples": sum(len(group.sample_ids) for group in duplicate_groups),
        "near_duplicate_leakage_rate": leakage.near_duplicate_leakage_rate,
        "physical_entities": len({sample.entity_id for sample in samples}),
        "positive_samples": sum(bool(sample.response.objects) for sample in samples),
        "negative_samples": sum(not sample.response.objects for sample in samples),
        "test_positive_samples": split_results[("test", "violation")],
        "test_negative_samples": split_results[("test", "compliant")],
        "test_hard_negative_count": test_hard_negative_count,
        "test_hard_negative_rate": test_hard_negative_count / test_count if test_count else 0.0,
        "evaluation_status": "formal" if formal_kpi_eligible else "pilot",
        "formal_kpi_eligible": formal_kpi_eligible,
        "minimum_formal_kpi_test_samples": minimum_formal_kpi_test_samples,
        "formal_kpi_ineligibility_reason": (
            None
            if formal_kpi_eligible
            else f"test split has {test_count} samples; minimum is {minimum_formal_kpi_test_samples}"
        ),
    }
    manifest = replace(manifest, statistics=dict(manifest.statistics) | extra_statistics)
    statistics = dict(statistics) | extra_statistics
    probes = _select_probes(samples, assignments)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    probe_entries = []
    for sample_id in probes:
        sample = sample_by_id[sample_id]
        source_image = destination / sample.image
        probe_image = destination / "probes" / f"{sample.sample_id}{source_image.suffix.lower()}"
        probe_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, probe_image)
        probe_entries.append(
            {
                "sample_id": sample.sample_id,
                "probe_image": probe_image.relative_to(destination).as_posix(),
                "source_image": sample.image,
                "sha256": sha256_file(probe_image),
                "result": sample.response.result,
                "objects": [item.to_dict() for item in sample.response.objects],
                "license_id": sample.license.identifier,
                "attribution": sample.license.attribution,
            }
        )

    _atomic_text(destination / "samples.jsonl", _jsonl(samples))
    _atomic_text(
        destination / "evaluation_ground_truth.jsonl",
        _evaluation_ground_truth_jsonl(samples, assignments, validation),
    )
    manifest.write_atomic(destination / "manifest.json")
    manifest_hash = sha256_file(destination / "manifest.json")
    _atomic_text(destination / "manifest.sha256", f"{manifest_hash}  manifest.json\n")
    _atomic_text(destination / "source_record.json", _json(source_record.to_dict()))
    _atomic_text(destination / "leakage_report.json", _json(leakage.to_dict()))
    _atomic_text(destination / "probe_manifest.json", _json({"probes": probe_entries}))
    _atomic_text(
        destination / "dataset_summary.json",
        _json(
            {
                "dataset_version": dataset_version,
                "manifest_sha256": manifest_hash,
                "statistics": statistics,
                "exact_duplicate_removed_ids": list(removed_ids),
                "probe_sample_ids": list(probes),
                "fixture_or_mock": False,
                "model_metrics": None,
                "model_metrics_note": "No model was run; dataset statistics are not model scores.",
            }
        ),
    )
    card = generate_data_card(
        manifest,
        statistics,
        title="KolektorSDD V0 Industrial Defect Localization Data Card",
        pilot_only=not formal_kpi_eligible,
    )
    _atomic_text(destination / "DATA_CARD.md", card)
    return KSDDPreparationResult(
        samples=samples,
        assignments=assignments,
        duplicate_groups=duplicate_groups,
        exact_duplicate_removed_ids=removed_ids,
        validation=validation,
        leakage=leakage,
        manifest=manifest,
        statistics=statistics,
        probe_sample_ids=probes,
    )
