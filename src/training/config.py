"""Load and validate versioned LoRA/QLoRA experiment configuration."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import AdaptationMethod, AdapterTrainingConfig, ModuleStrategy


def load_training_config(path: str | Path) -> AdapterTrainingConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config root must be a JSON object")
    required = {
        "base_model_id",
        "method",
        "module_strategy",
        "rank",
        "alpha",
        "dropout",
        "seed",
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "learning_rate",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"missing training config fields: {', '.join(missing)}")
    allowed = required | {
        "schema_version",
        "gradient_checkpointing",
        "quantization_bits",
        "extra",
    }
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"unknown training config fields: {', '.join(unknown)}")
    if payload.get("schema_version", "1.0") != "1.0":
        raise ValueError("unsupported training config schema_version")
    gradient_checkpointing = payload.get("gradient_checkpointing", True)
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError("gradient_checkpointing must be a boolean")
    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("extra must be a JSON object")
    for field_name in ("base_model_id", "method", "module_strategy"):
        if not isinstance(payload[field_name], str):
            raise ValueError(f"{field_name} must be a string")
    for field_name in (
        "rank",
        "alpha",
        "seed",
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
    ):
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
    for field_name in ("dropout", "learning_rate"):
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be numeric")
    quantization_bits = payload.get("quantization_bits")
    if quantization_bits is not None and (
        isinstance(quantization_bits, bool) or not isinstance(quantization_bits, int)
    ):
        raise ValueError("quantization_bits must be an integer or null")
    return AdapterTrainingConfig(
        base_model_id=str(payload["base_model_id"]),
        method=AdaptationMethod(payload["method"]),
        module_strategy=ModuleStrategy(payload["module_strategy"]),
        rank=payload["rank"],
        alpha=payload["alpha"],
        dropout=payload["dropout"],
        seed=payload["seed"],
        batch_size=payload["batch_size"],
        gradient_accumulation_steps=payload["gradient_accumulation_steps"],
        max_steps=payload["max_steps"],
        learning_rate=payload["learning_rate"],
        gradient_checkpointing=gradient_checkpointing,
        quantization_bits=quantization_bits,
        extra=extra,
    )
