"""Controlled local QLoRA runner built on the official MLX-VLM trainer API."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from src.inference.cache import resolve_cached_model_path
from src.inference.config import load_model_config
from src.inference.performance import process_peak_memory_mb
from src.inference.real_probe import write_json_atomic

from .config import load_training_config
from .contracts import AdaptationMethod, ModuleStrategy

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_CONFIG = ROOT / "configs/models/ksdd_qwen3_vl_2b_qlora_phase6.json"
DEFAULT_MODEL_CONFIG = ROOT / "configs/models/qwen3_vl_2b_mlx_4bit.json"
DEFAULT_SFT_ROOT = ROOT / "artifacts/model/phase6/ksdd_sft_v1"
CONTRACT_TRANSFORM = "model_adapter_refusal_code_v1"


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def normalize_sft_record_for_adapter(record: Mapping[str, Any]) -> dict[str, Any]:
    """Add the non-refusal field required by ModelAdapter without mutating source data."""

    # datasets>=5 supplies a LazyRow Mapping that is deliberately not JSON
    # serializable. Copy its materialized values before changing nested text.
    normalized = copy.deepcopy(dict(record))
    try:
        assistant_text = normalized["messages"][-1]["content"][0]["text"]
        answer = json.loads(assistant_text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("SFT record does not contain one valid assistant JSON answer") from exc
    if not isinstance(answer, dict):
        raise ValueError("SFT assistant answer must be a JSON object")
    if answer.get("uncertain") is not False:
        raise ValueError("controlled KSDD SFT record must have uncertain=false")
    if "refusal_code" in answer and answer["refusal_code"] is not None:
        raise ValueError("non-uncertain SFT answer cannot contain a refusal code")
    answer["refusal_code"] = None
    normalized["messages"][-1]["content"][0]["text"] = json.dumps(
        answer, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return normalized


def validate_sft_package(sft_root: str | Path) -> dict[str, Any]:
    root = Path(sft_root).resolve()
    manifest_path = root / "sft_manifest.json"
    manifest = _read_json_object(manifest_path)
    if manifest.get("source_manifest_sha256") != (
        "fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a"
    ):
        raise ValueError("SFT package is not derived from the frozen KSDD V0 manifest")
    statistics = manifest.get("statistics")
    split_counts = statistics.get("split_counts") if isinstance(statistics, Mapping) else None
    if (
        not isinstance(split_counts, Mapping)
        or set(split_counts) != {"train", "valid", "test"}
        or any(not isinstance(split_counts[split], int) or split_counts[split] <= 0 for split in split_counts)
    ):
        raise ValueError("SFT package split counts are invalid")
    file_hashes = manifest.get("file_sha256")
    if not isinstance(file_hashes, Mapping):
        raise ValueError("SFT manifest file_sha256 is missing")
    for split in ("train", "valid", "test"):
        relative = f"hf/{split}.jsonl"
        path = root / relative
        if not path.is_file() or _sha256(path) != file_hashes.get(relative):
            raise ValueError(f"SFT file hash mismatch: {relative}")
        if _count_jsonl_rows(path) != split_counts[split]:
            raise ValueError(f"SFT row count mismatch: {relative}")
    leakage_path = root / "leakage_report.json"
    leakage = (
        _read_json_object(leakage_path)
        if leakage_path.is_file()
        else manifest.get("leakage")
    )
    if not isinstance(leakage, Mapping):
        raise ValueError("SFT leakage evidence is missing")
    if leakage.get("passes") is not True:
        raise ValueError("SFT package leakage gate did not pass")
    overlap_fields = [field for field in leakage if "overlap" in field]
    if not overlap_fields:
        raise ValueError("SFT leakage evidence has no overlap fields")
    for field in overlap_fields:
        if leakage[field] != []:
            raise ValueError(f"SFT leakage field is not empty: {field}")
    return manifest


def run_training(
    training_config_path: str | Path,
    model_config_path: str | Path,
    sft_root: str | Path,
    output_dir: str | Path,
    *,
    steps: int,
    run_kind: str,
    validation_batches: int,
) -> dict[str, Any]:
    if run_kind not in {"probe", "smoke", "formal"}:
        raise ValueError("run_kind must be probe, smoke, or formal")
    training_config = load_training_config(training_config_path)
    model_config = load_model_config(model_config_path)
    if training_config.method is not AdaptationMethod.QLORA:
        raise ValueError("phase 6 runner requires QLoRA")
    if training_config.module_strategy is not ModuleStrategy.LANGUAGE_ONLY:
        raise ValueError("phase 6 runner trains language modules only")
    if model_config.quantization.bits != training_config.quantization_bits:
        raise ValueError("base model quantization differs from training config")
    if steps <= 0 or steps > training_config.max_steps:
        raise ValueError("steps must be positive and no greater than configured max_steps")
    if run_kind == "probe" and steps != 1:
        raise ValueError("probe run must contain exactly one step")
    if run_kind == "smoke" and not 10 <= steps <= 20:
        raise ValueError("smoke run must contain 10-20 steps")
    if run_kind == "formal" and steps != training_config.max_steps:
        raise ValueError("formal run must use the configured max_steps")
    if validation_batches <= 0:
        raise ValueError("validation_batches must be positive")

    sft_manifest = validate_sft_package(sft_root)
    source = resolve_cached_model_path(model_config)
    if source is None:
        raise RuntimeError("pinned Qwen3-VL snapshot is not available locally")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite training run: {destination}")
    destination.mkdir(parents=True)
    log_path = destination / "training.log"
    manifest_path = destination / "run_manifest.json"
    started_at = _now()
    started = time.perf_counter()
    status = "failed"
    error: dict[str, str] | None = None
    adapter_path = destination / "adapter" / "adapters.safetensors"
    memory_limit_mb = int(training_config.extra.get("memory_limit_mb", 12288))
    max_seq_length = int(training_config.extra.get("max_seq_length", 768))
    image_resize_size = int(training_config.extra.get("image_resize_size", 0))

    with log_path.open("w", encoding="utf-8") as log_stream:
        tee_stdout = Tee(sys.stdout, log_stream)
        tee_stderr = Tee(sys.stderr, log_stream)
        try:
            with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                _train_official(
                    source,
                    Path(sft_root).resolve(),
                    training_config,
                    adapter_path,
                    steps=steps,
                    validation_batches=validation_batches,
                    max_seq_length=max_seq_length,
                    image_resize_size=image_resize_size,
                    memory_limit_mb=memory_limit_mb,
                )
            status = "completed"
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                print(json.dumps({"training_error": error}, ensure_ascii=False), file=sys.stderr)

    wall_seconds = round(time.perf_counter() - started, 3)
    peak_mlx_mb = _mlx_peak_memory_mb()
    peak_process_mb = process_peak_memory_mb()
    runtime_error = error
    if peak_mlx_mb is not None and peak_mlx_mb > memory_limit_mb:
        status = "resource_limit_exceeded"
        error = {
            "type": "MemoryLimitExceeded",
            "message": f"MLX peak {peak_mlx_mb} MB exceeded limit {memory_limit_mb} MB",
        }
    adapter_hash = _sha256(adapter_path) if adapter_path.is_file() else None
    adapter_config = adapter_path.parent / "adapter_config.json"
    run_manifest = {
        "schema_version": "1.0",
        "run_kind": run_kind,
        "status": status,
        "started_at": started_at,
        "ended_at": _now(),
        "error": error,
        "model": {
            "model_id": model_config.model_id,
            "base_model_id": model_config.base_model_id,
            "revision": model_config.revision,
            "quantization_bits": model_config.quantization.bits,
            "cached_source": str(source),
            "weight_sha256": _sha256(source / "model.safetensors"),
        },
        "dataset": {
            "root": str(Path(sft_root).resolve()),
            "dataset_version": sft_manifest.get("dataset_version"),
            "sft_manifest_sha256": _sha256(Path(sft_root).resolve() / "sft_manifest.json"),
            "source_manifest_sha256": sft_manifest.get("source_manifest_sha256"),
            "train_sha256": sft_manifest["file_sha256"]["hf/train.jsonl"],
            "validation_sha256": sft_manifest["file_sha256"]["hf/valid.jsonl"],
            "contract_transform": CONTRACT_TRANSFORM,
        },
        "training": {
            "method": training_config.method.value,
            "module_strategy": training_config.module_strategy.value,
            "rank": training_config.rank,
            "alpha": training_config.alpha,
            "dropout": training_config.dropout,
            "seed": training_config.seed,
            "batch_size": training_config.batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "effective_batch_size": (
                training_config.batch_size * training_config.gradient_accumulation_steps
            ),
            "steps": steps,
            "planned_optimizer_updates_floor": (
                steps // training_config.gradient_accumulation_steps
            ),
            "learning_rate": training_config.learning_rate,
            "gradient_checkpointing": training_config.gradient_checkpointing,
            "train_on_completions": True,
            "max_seq_length": max_seq_length,
            "max_output_tokens": int(training_config.extra.get("max_output_tokens", 128)),
            "image_resize_size": image_resize_size or None,
            "vision_encoder_frozen": bool(
                training_config.extra.get("vision_encoder_frozen", True)
            ),
            "validation_batches": validation_batches,
        },
        "resources": {
            "wall_seconds": wall_seconds,
            "mlx_peak_allocated_mb": peak_mlx_mb,
            "process_peak_rss_mb": peak_process_mb,
            "memory_limit_mb": memory_limit_mb,
            "max_wall_minutes": int(training_config.extra.get("max_wall_minutes", 60)),
            "memory_scope": "mlx_allocator_peak_and_process_peak_rss",
            "runtime_error": runtime_error,
        },
        "artifacts": {
            "adapter_path": str(adapter_path) if adapter_path.is_file() else None,
            "adapter_sha256": adapter_hash,
            "adapter_config_path": str(adapter_config) if adapter_config.is_file() else None,
            "adapter_config_sha256": _sha256(adapter_config) if adapter_config.is_file() else None,
            "training_log": str(log_path),
            "training_log_sha256": _sha256(log_path),
            "resolved_training_config": str(destination / "resolved_training_config.json"),
        },
    }
    shutil.copyfile(training_config_path, destination / "resolved_training_config.json")
    run_manifest["artifacts"]["resolved_training_config_sha256"] = _sha256(
        destination / "resolved_training_config.json"
    )
    write_json_atomic(run_manifest, manifest_path)
    if status != "completed":
        raise RuntimeError(f"controlled training did not complete: {error}")
    return run_manifest


def _train_official(
    model_source: Path,
    sft_root: Path,
    config: Any,
    adapter_path: Path,
    *,
    steps: int,
    validation_batches: int,
    max_seq_length: int,
    image_resize_size: int,
    memory_limit_mb: int,
) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from datasets import load_dataset
    from mlx_vlm import load
    from mlx_vlm.lora import setup_model_for_training
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
    from mlx_vlm.trainer.utils import not_supported_for_training, print_trainable_parameters

    random.seed(config.seed)
    np.random.seed(config.seed)
    mx.random.seed(config.seed)
    mx.reset_peak_memory()
    mx.set_memory_limit(memory_limit_mb * 1024 * 1024)
    print(f"Loading pinned local model: {model_source}")
    model, processor = load(str(model_source), trust_remote_code=False)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in not_supported_for_training:
        raise ValueError(f"model type is not supported for training: {model_type}")

    data_files = {
        "train": str(sft_root / "hf/train.jsonl"),
        "validation": str(sft_root / "hf/valid.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(normalize_sft_record_for_adapter)
    print(
        f"Loaded SFT dataset: train={len(dataset['train'])}, "
        f"validation={len(dataset['validation'])}, transform={CONTRACT_TRANSFORM}"
    )
    model_dict = model.config.__dict__
    image_resize_shape = (
        (image_resize_size, image_resize_size) if image_resize_size > 0 else None
    )
    train_dataset = VisionDataset(
        dataset["train"],
        model_dict,
        processor,
        train_on_completions=True,
        image_resize_shape=image_resize_shape,
    )
    validation_dataset = VisionDataset(
        dataset["validation"],
        model_dict,
        processor,
        train_on_completions=True,
        image_resize_shape=image_resize_shape,
    )

    class SetupArgs:
        full_finetune = False
        train_vision = False
        lora_rank = config.rank
        lora_alpha = config.alpha
        lora_dropout = config.dropout

    model = setup_model_for_training(model, SetupArgs())
    print_trainable_parameters(model)
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    args = TrainingArgs(
        batch_size=config.batch_size,
        iters=steps,
        val_batches=validation_batches,
        steps_per_report=max(1, min(10, steps)),
        steps_per_eval=steps,
        steps_per_save=steps,
        max_seq_length=max_seq_length,
        adapter_file=str(adapter_path),
        grad_checkpoint=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        grad_clip=1.0,
        warmup_steps=min(10, steps),
        min_learning_rate=config.learning_rate / 10.0,
        full_finetune=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        args=args,
        train_on_completions=True,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mlx_peak_memory_mb() -> float | None:
    try:
        import mlx.core as mx

        return round(float(mx.get_peak_memory()) / (1024.0 * 1024.0), 3)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--run-kind", choices=("probe", "smoke", "formal"), required=True)
    parser.add_argument("--validation-batches", type=int, default=2)
    arguments = parser.parse_args(argv)
    report = run_training(
        arguments.training_config,
        arguments.model_config,
        arguments.sft_root,
        arguments.output_dir,
        steps=arguments.steps,
        run_kind=arguments.run_kind,
        validation_batches=arguments.validation_batches,
    )
    print(json.dumps({"status": report["status"], **report["resources"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
