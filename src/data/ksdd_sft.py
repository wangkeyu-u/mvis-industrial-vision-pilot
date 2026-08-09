"""Deterministic MLX-VLM SFT export from the frozen KSDD entity split."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .deduplication import find_near_duplicate_groups, sha256_file
from .leakage import detect_split_leakage
from .manifest import DatasetManifest
from .pipeline import load_samples_jsonl
from .schema import DataSample, SchemaError

DEFAULT_DATASET_VERSION = "ksdd_sft-1.0.0"
DEFAULT_PROMPT_VERSION = "ksdd_audit_prompt-1.0.0"
TEST_PROMPT_VARIANT = "bilingual_audit_direct"
TRAIN_PROMPT_VARIANTS = (
    "bilingual_audit_direct",
    "bilingual_audit_quality_gate",
    "bilingual_audit_evidence_first",
)
VERSION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*-\d+\.\d+\.\d+$")

PROMPTS = {
    "bilingual_audit_direct": (
        "Inspect the electrical-commutator surface for any visible production defect. "
        "Return exactly one JSON object and no Markdown. Use result=violation with one or "
        "more surface_defect objects and original-image integer bbox [x1,y1,x2,y2] when a "
        "defect is visible; otherwise use result=compliant and objects=[]. Set uncertain=false. "
        "/ 检查该电机换向器表面是否存在可见生产缺陷。只返回一个严格 JSON 对象，不要输出 "
        "Markdown。若有缺陷，result=violation，并用一个或多个 surface_defect 对象及原图整数坐标 "
        "bbox [x1,y1,x2,y2] 定位；若无缺陷，result=compliant 且 objects=[]。uncertain 必须为 false。"
    ),
    "bilingual_audit_quality_gate": (
        "Act as an industrial visual quality gate for this commutator surface. Decide whether a "
        "visible surface anomaly violates quality requirements and ground every violation in "
        "original-image integer xyxy coordinates. Output only one strict JSON object: compliant "
        "requires objects=[], violation requires surface_defect bbox evidence, and uncertain=false. "
        "/ 作为工业视觉质检门禁，判断该换向器表面的可见异常是否违反质量要求，并以原图整数 xyxy "
        "坐标定位每处违规。仅输出一个严格 JSON 对象：compliant 必须 objects=[]，violation 必须包含 "
        "surface_defect bbox 证据，且 uncertain=false。"
    ),
    "bilingual_audit_evidence_first": (
        "Examine the full commutator image, locate visible surface-defect evidence first, then issue "
        "the compliance decision. Return JSON only, without code fences or commentary. A positive "
        "answer is result=violation with surface_defect bbox [x1,y1,x2,y2] in original pixels; a "
        "negative answer is result=compliant with objects=[]; uncertain=false. / 先检查整张换向器图像并定位"
        "可见表面缺陷证据，再给出合规结论。只返回 JSON，不要代码围栏或额外说明。正例为 "
        "result=violation，并给出原图像素 surface_defect bbox [x1,y1,x2,y2]；负例为 "
        "result=compliant 且 objects=[]；uncertain=false。"
    ),
}


@dataclass(frozen=True)
class KSDDSFTExportResult:
    directory: Path
    dataset_version: str
    prompt_version: str
    source_manifest_sha256: str
    split_counts: Mapping[str, int]
    file_sha256: Mapping[str, str]
    leakage_report: Mapping[str, Any]
    manifest_sha256: str


def _is_bilingual(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text)) and bool(re.search(r"[\u4e00-\u9fff]", text))


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _validate_version(value: str, field_name: str) -> None:
    if not VERSION_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match name-major.minor.patch")


def _source_to_sft_split(source_split: str) -> str:
    if source_split == "validation":
        return "valid"
    if source_split in {"train", "test"}:
        return source_split
    raise ValueError(f"unsupported source split: {source_split!r}")


def prompt_variant_for(sample_id: str, split: str) -> str:
    """Choose one semantic-equivalent train prompt; valid/test stay fixed."""

    if split != "train":
        return TEST_PROMPT_VARIANT
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return TRAIN_PROMPT_VARIANTS[int.from_bytes(digest[:2], "big") % len(TRAIN_PROMPT_VARIANTS)]


def build_sft_answer(sample: DataSample) -> dict[str, Any]:
    """Build the strict bilingual answer solely from the frozen annotation."""

    positive = bool(sample.response.objects)
    result = "violation" if positive else "compliant"
    if sample.response.result != result:
        raise SchemaError(
            f"sample {sample.sample_id} has inconsistent result and frozen object annotation"
        )
    return {
        "result": result,
        "objects": [item.to_dict() for item in sample.response.objects],
        "reason": (
            "A visible surface anomaly is present in the localized region. / "
            "定位区域存在可见表面异常。"
            if positive
            else "No visible surface defect is present. / 未发现可见表面缺陷。"
        ),
        "uncertain": False,
    }


def build_sft_record(
    sample: DataSample,
    *,
    source_split: str,
    image_reference: str,
    dataset_version: str,
    source_dataset_version: str,
    prompt_version: str,
) -> dict[str, Any]:
    split = _source_to_sft_split(source_split)
    variant = prompt_variant_for(sample.sample_id, split)
    prompt = PROMPTS[variant]
    answer = _compact_json(build_sft_answer(sample))
    return {
        "sample_id": sample.sample_id,
        "entity_id": sample.entity_id,
        "split": split,
        "dataset_version": dataset_version,
        "source_dataset_version": source_dataset_version,
        "prompt_version": prompt_version,
        "prompt_variant": variant,
        "images": [image_reference],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_reference},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }


def _answer_from_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        raw = record["messages"][1]["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SchemaError("SFT record does not contain one assistant text answer") from exc
    if not isinstance(raw, str) or not raw.startswith("{") or not raw.endswith("}"):
        raise SchemaError("assistant answer must be exactly one JSON object without fences")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"assistant answer is invalid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise SchemaError("assistant answer must decode to an object")
    return value


def validate_sft_record(
    record: Mapping[str, Any],
    truth: DataSample,
    image_path: str | Path,
    *,
    expected_dataset_version: str,
    expected_source_dataset_version: str,
    expected_prompt_version: str,
) -> None:
    """Validate the MLX-VLM message contract and its frozen truth alignment."""

    required = {
        "sample_id",
        "entity_id",
        "split",
        "dataset_version",
        "source_dataset_version",
        "prompt_version",
        "prompt_variant",
        "images",
        "messages",
    }
    if set(record) != required:
        raise SchemaError(f"SFT record fields must exactly equal {sorted(required)}")
    if record["sample_id"] != truth.sample_id or record["entity_id"] != truth.entity_id:
        raise SchemaError("SFT identity differs from frozen source sample")
    if record["dataset_version"] != expected_dataset_version:
        raise SchemaError("unexpected SFT dataset version")
    if record["source_dataset_version"] != expected_source_dataset_version:
        raise SchemaError("unexpected source dataset version")
    if record["prompt_version"] != expected_prompt_version:
        raise SchemaError("unexpected prompt version")
    split = record["split"]
    if split not in {"train", "valid", "test"}:
        raise SchemaError("SFT split must be train, valid, or test")
    variant = record["prompt_variant"]
    expected_variant = prompt_variant_for(truth.sample_id, split)
    if variant != expected_variant or variant not in PROMPTS:
        raise SchemaError("prompt variant violates deterministic split policy")
    if split == "test" and variant != TEST_PROMPT_VARIANT:
        raise SchemaError("test prompt must use the single frozen prompt variant")

    images = record["images"]
    messages = record["messages"]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise SchemaError("images must contain exactly one path string")
    if not isinstance(messages, list) or len(messages) != 2:
        raise SchemaError("messages must contain exactly one user and one assistant turn")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise SchemaError("SFT message roles must be user then assistant")
    content = user.get("content")
    if not isinstance(content, list) or len(content) != 2:
        raise SchemaError("user content must contain image then text")
    if content[0] != {"type": "image", "image": images[0]}:
        raise SchemaError("user image content must match the images column")
    if content[1] != {"type": "text", "text": PROMPTS[variant]}:
        raise SchemaError("user prompt text does not match the frozen prompt catalog")
    if not _is_bilingual(content[1]["text"]):
        raise SchemaError("user prompt must contain both English and Chinese")

    answer = _answer_from_record(record)
    if set(answer) != {"result", "objects", "reason", "uncertain"}:
        raise SchemaError("answer must contain only result, objects, reason, and uncertain")
    expected_answer = build_sft_answer(truth)
    if answer != expected_answer:
        raise SchemaError("assistant answer differs from the frozen annotation")
    if not _is_bilingual(str(answer["reason"])):
        raise SchemaError("assistant reason must contain both English and Chinese")
    if answer["uncertain"] is not False:
        raise SchemaError("SFT answer uncertain must be false")
    if answer["result"] == "compliant" and answer["objects"] != []:
        raise SchemaError("compliant answer must use objects=[]")
    if answer["result"] == "violation" and not answer["objects"]:
        raise SchemaError("violation answer must contain bbox evidence")
    with Image.open(image_path) as image:
        width, height = image.size
    for item in answer["objects"]:
        bbox = item["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
            or bbox[0] < 0
            or bbox[1] < 0
            or bbox[0] >= bbox[2]
            or bbox[1] >= bbox[3]
            or bbox[2] > width
            or bbox[3] > height
        ):
            raise SchemaError(f"answer bbox is invalid for {width}x{height}: {bbox}")


def load_sft_jsonl(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"invalid SFT JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, Mapping):
                raise SchemaError(f"SFT JSONL line {line_number} must be an object")
            records.append(value)
    return tuple(records)


def _image_reference(final_path: Path, reference_root: Path) -> str:
    try:
        return final_path.relative_to(reference_root.resolve()).as_posix()
    except ValueError:
        return str(final_path)


def _evaluation_truth(
    records: Sequence[Mapping[str, Any]],
    samples: Mapping[str, DataSample],
    copied_images: Mapping[str, Path],
    prompt_version: str,
) -> str:
    rows = []
    for record in records:
        if record["split"] != "test":
            continue
        sample = samples[str(record["sample_id"])]
        with Image.open(copied_images[sample.sample_id]) as image:
            width, height = image.size
        rows.append(
            {
                "sample_id": sample.sample_id,
                "response": build_sft_answer(sample),
                "image_width": width,
                "image_height": height,
                "difficulty": list(sample.difficulty),
                "source": "ksdd_sft_v1",
                "slices": [
                    "dataset:ksdd_sft_v1",
                    f"prompt_version:{prompt_version}",
                    f"prompt_variant:{TEST_PROMPT_VARIANT}",
                ],
            }
        )
    return "".join(_compact_json(row) + "\n" for row in rows)


def export_ksdd_sft(
    source_root: str | Path,
    output_root: str | Path,
    *,
    expected_source_manifest_sha256: str,
    created_at: str,
    record_schema_path: str | Path,
    answer_schema_path: str | Path,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    reference_root: str | Path | None = None,
) -> KSDDSFTExportResult:
    """Freeze a self-contained, source-split-preserving KSDD SFT package."""

    _validate_version(dataset_version, "dataset_version")
    _validate_version(prompt_version, "prompt_version")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if parsed_created_at.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_manifest_sha256):
        raise ValueError("expected_source_manifest_sha256 must be lowercase SHA-256")

    source = Path(source_root).resolve()
    destination = Path(output_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset root does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite SFT dataset: {destination}")
    manifest_path = source / "manifest.json"
    samples_path = source / "samples.jsonl"
    actual_source_manifest_sha256 = sha256_file(manifest_path)
    if actual_source_manifest_sha256 != expected_source_manifest_sha256:
        raise SchemaError(
            "source manifest hash mismatch; refusing to export from an unfrozen KSDD version"
        )
    source_manifest = DatasetManifest.load(manifest_path)
    if not source_manifest.frozen_test:
        raise SchemaError("source KSDD test split must be frozen")
    samples = load_samples_jsonl(samples_path)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    entry_by_id = {entry.sample_id: entry for entry in source_manifest.entries}
    if set(sample_by_id) != set(entry_by_id):
        raise SchemaError("source samples and manifest entries differ")

    assignments = {entry.sample_id: entry.split for entry in source_manifest.entries}
    max_distance = int(source_manifest.split_strategy.get("max_hamming_distance", 5))
    duplicate_groups = find_near_duplicate_groups(
        {entry.sample_id: entry.perceptual_hash for entry in source_manifest.entries},
        max_distance=max_distance,
    )
    source_leakage = detect_split_leakage(samples, assignments, duplicate_groups)
    if not source_leakage.passes(0.0):
        raise SchemaError(f"source leakage gate failed: {source_leakage.to_dict()}")

    staging_parent = destination.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=staging_parent))
    final_reference_root = (
        Path(reference_root).resolve() if reference_root is not None else Path.cwd().resolve()
    )
    try:
        copied_images: dict[str, Path] = {}
        records_by_split: dict[str, list[Mapping[str, Any]]] = {
            "train": [],
            "valid": [],
            "test": [],
        }
        for entry in source_manifest.entries:
            sample = sample_by_id[entry.sample_id]
            source_image = source / entry.image
            if sha256_file(source_image) != entry.sha256:
                raise SchemaError(f"source image hash changed: {entry.sample_id}")
            relative_image = Path(entry.image)
            staged_image = staging / relative_image
            staged_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, staged_image)
            copied_images[sample.sample_id] = staged_image
            final_image = destination / relative_image
            image_reference = _image_reference(final_image, final_reference_root)
            record = build_sft_record(
                sample,
                source_split=entry.split,
                image_reference=image_reference,
                dataset_version=dataset_version,
                source_dataset_version=source_manifest.dataset_version,
                prompt_version=prompt_version,
            )
            validate_sft_record(
                record,
                sample,
                staged_image,
                expected_dataset_version=dataset_version,
                expected_source_dataset_version=source_manifest.dataset_version,
                expected_prompt_version=prompt_version,
            )
            records_by_split[str(record["split"])].append(record)

        for split, records in records_by_split.items():
            _write_text(
                staging / "hf" / f"{split}.jsonl",
                "".join(_compact_json(record) + "\n" for record in records),
            )

        record_schema = Path(record_schema_path).resolve()
        answer_schema = Path(answer_schema_path).resolve()
        if not record_schema.is_file() or not answer_schema.is_file():
            raise FileNotFoundError("SFT record and answer schema files are required")
        (staging / "schema").mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_schema, staging / "schema" / record_schema.name)
        shutil.copy2(answer_schema, staging / "schema" / answer_schema.name)

        all_records = [record for values in records_by_split.values() for record in values]
        _write_text(
            staging / "evaluation_ground_truth.jsonl",
            _evaluation_truth(all_records, sample_by_id, copied_images, prompt_version),
        )

        source_ids = {
            split: {entry.sample_id for entry in source_manifest.entries if entry.split == split}
            for split in ("train", "validation", "test")
        }
        source_entities = {
            split: {
                entry.entity_id for entry in source_manifest.entries if entry.split == split
            }
            for split in ("train", "validation", "test")
        }
        source_hashes = {
            split: {entry.sha256 for entry in source_manifest.entries if entry.split == split}
            for split in ("train", "validation", "test")
        }
        leakage_report = {
            "source_manifest_sha256": actual_source_manifest_sha256,
            "source_report": source_leakage.to_dict(),
            "train_test_sample_overlap": sorted(source_ids["train"] & source_ids["test"]),
            "train_test_entity_overlap": sorted(
                source_entities["train"] & source_entities["test"]
            ),
            "train_test_image_sha256_overlap": sorted(
                source_hashes["train"] & source_hashes["test"]
            ),
            "test_derived_training_sample_count": 0,
            "source_assignments_preserved": True,
            "test_prompt_fixed": True,
            "passes": True,
        }
        if any(
            leakage_report[key]
            for key in (
                "train_test_sample_overlap",
                "train_test_entity_overlap",
                "train_test_image_sha256_overlap",
            )
        ):
            raise SchemaError(f"SFT train/test isolation failed: {leakage_report}")
        _write_json(staging / "leakage_report.json", leakage_report)

        prompt_counts = Counter(str(record["prompt_variant"]) for record in all_records)
        prompt_counts_by_split = {
            split: dict(
                sorted(Counter(str(record["prompt_variant"]) for record in records).items())
            )
            for split, records in records_by_split.items()
        }
        split_counts = {split: len(records) for split, records in records_by_split.items()}
        positive_counts = {
            split: sum(
                _answer_from_record(record)["result"] == "violation" for record in records
            )
            for split, records in records_by_split.items()
        }
        negative_counts = {
            split: len(records_by_split[split]) - positive_counts[split]
            for split in records_by_split
        }
        prompt_catalog = {
            "prompt_version": prompt_version,
            "train_variants": {
                name: {"text": PROMPTS[name], "sha256": _sha256_text(PROMPTS[name])}
                for name in TRAIN_PROMPT_VARIANTS
            },
            "valid_variant": TEST_PROMPT_VARIANT,
            "test_variant": TEST_PROMPT_VARIANT,
            "test_prompt_fixed": True,
            "test_prompt_sha256": _sha256_text(PROMPTS[TEST_PROMPT_VARIANT]),
        }
        _write_json(staging / "prompt_catalog.json", prompt_catalog)

        readiness_report = {
            "report_type": "ksdd_sft_fair_evaluation_readiness",
            "dataset_version": dataset_version,
            "prompt_version": prompt_version,
            "pilot_only": True,
            "formal_kpi_eligible": False,
            "pilot_reason": (
                f"frozen test split has {split_counts['test']} samples; formal KPI floor is 300"
            ),
            "model_run": False,
            "zero_shot_status": "not_run",
            "lora_status": "not_run",
            "metrics": {
                "zero_shot": None,
                "lora": None,
                "comparison": None,
            },
            "fairness_requirements": {
                "same_test_sample_ids": True,
                "same_test_prompt_version": prompt_version,
                "same_test_prompt_variant": TEST_PROMPT_VARIANT,
                "same_generation_config_required": True,
                "only_difference": "LoRA adapter enabled versus disabled",
            },
            "evaluation_command": (
                ".venv/bin/python -m src.evaluation.sft_fair_cli "
                "--ground-truth data/processed/ksdd_sft_v1/evaluation_ground_truth.jsonl "
                "--zero-shot-predictions ZERO_SHOT.jsonl --lora-predictions LORA.jsonl "
                "--zero-shot-run-config ZERO_SHOT_RUN.json "
                "--lora-run-config LORA_RUN.json "
                "--data-manifest data/processed/ksdd_sft_v1/sft_manifest.json "
                "--output-dir EVAL_OUTPUT --created-at ISO8601"
            ),
        }
        _write_json(staging / "pilot_readiness_report.json", readiness_report)

        file_hashes = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        test_prompt_variants = {
            str(record["prompt_variant"]) for record in records_by_split["test"]
        }
        if test_prompt_variants != {TEST_PROMPT_VARIANT}:
            raise SchemaError("test split must contain exactly one frozen prompt variant")
        manifest = {
            "schema_version": "1.0.0",
            "dataset_version": dataset_version,
            "source_dataset_version": source_manifest.dataset_version,
            "source_manifest_sha256": actual_source_manifest_sha256,
            "created_at": created_at,
            "prompt_version": prompt_version,
            "prompt_policy": {
                "train_variants": list(TRAIN_PROMPT_VARIANTS),
                "valid_variant": TEST_PROMPT_VARIANT,
                "test_variant": TEST_PROMPT_VARIANT,
                "test_prompt_fixed": True,
                "test_prompt_sha256": _sha256_text(PROMPTS[TEST_PROMPT_VARIANT]),
            },
            "format": {
                "name": "mlx-vlm images/messages",
                "verified_against": "mlx-vlm 0.6.10 official LoRA contract",
                "hf_data_directory": "hf",
                "train_file": "hf/train.jsonl",
                "validation_file": "hf/valid.jsonl",
                "test_file": "hf/test.jsonl",
            },
            "statistics": {
                "total_samples": len(all_records),
                "split_counts": split_counts,
                "positive_counts": positive_counts,
                "negative_counts": negative_counts,
                "prompt_variant_counts": dict(sorted(prompt_counts.items())),
                "prompt_variant_counts_by_split": prompt_counts_by_split,
                "entity_count": len({sample.entity_id for sample in samples}),
                "test_prompt_variant_count": len(test_prompt_variants),
                "strict_answer_json_validity": 1.0,
                "test_derived_training_sample_count": 0,
                "evaluation_status": "pilot",
                "formal_kpi_eligible": False,
                "formal_kpi_ineligibility_reason": (
                    f"test split has {split_counts['test']} samples; minimum is 300"
                ),
            },
            "license": source_manifest.licenses["cc-by-nc-sa-4.0"].to_dict(),
            "file_sha256": file_hashes,
            "leakage_report": "leakage_report.json",
            "pilot_readiness_report": "pilot_readiness_report.json",
            "model_metrics": None,
            "model_metrics_note": "No model or training process was run during SFT export.",
        }
        _write_json(staging / "sft_manifest.json", manifest)
        manifest_hash = sha256_file(staging / "sft_manifest.json")
        _write_text(staging / "sft_manifest.sha256", f"{manifest_hash}  sft_manifest.json\n")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return KSDDSFTExportResult(
        directory=destination,
        dataset_version=dataset_version,
        prompt_version=prompt_version,
        source_manifest_sha256=actual_source_manifest_sha256,
        split_counts=split_counts,
        file_sha256=file_hashes,
        leakage_report=leakage_report,
        manifest_sha256=manifest_hash,
    )
