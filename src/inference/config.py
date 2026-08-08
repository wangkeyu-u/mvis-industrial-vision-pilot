"""Versionable model configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .contracts import GenerationConfig


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    bits: int
    group_size: int
    mode: str = "affine"

    def __post_init__(self) -> None:
        if self.bits not in {4, 8, 16}:
            raise ValueError("quantization bits must be one of 4, 8, 16")
        if self.group_size <= 0:
            raise ValueError("quantization group_size must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    schema_version: str
    alias: str
    family: str
    backend: str
    model_id: str
    base_model_id: str
    revision: str
    allow_download: bool
    local_model_path: str | None
    trust_remote_code: bool
    adapter_path: str | None
    generation: GenerationConfig
    quantization: QuantizationConfig
    limits: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported model config schema_version: {self.schema_version}")
        for field_name in ("alias", "family", "backend", "model_id", "base_model_id", "revision"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "alias": self.alias,
            "family": self.family,
            "backend": self.backend,
            "model_id": self.model_id,
            "base_model_id": self.base_model_id,
            "revision": self.revision,
            "allow_download": self.allow_download,
            "local_model_path": self.local_model_path,
            "trust_remote_code": self.trust_remote_code,
            "adapter_path": self.adapter_path,
            "generation": {
                "seed": self.generation.seed,
                "do_sample": self.generation.do_sample,
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "max_tokens": self.generation.max_tokens,
            },
            "quantization": {
                "bits": self.quantization.bits,
                "group_size": self.quantization.group_size,
                "mode": self.quantization.mode,
            },
            "limits": dict(self.limits),
        }


def load_model_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model config root must be a JSON object")
    required = {
        "schema_version",
        "alias",
        "family",
        "backend",
        "model_id",
        "base_model_id",
        "revision",
        "allow_download",
        "trust_remote_code",
        "generation",
        "quantization",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"missing model config fields: {', '.join(missing)}")
    allowed = required | {"local_model_path", "adapter_path", "limits"}
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"unknown model config fields: {', '.join(unknown)}")
    if not isinstance(payload["allow_download"], bool):
        raise ValueError("allow_download must be a boolean")
    if not isinstance(payload["trust_remote_code"], bool):
        raise ValueError("trust_remote_code must be a boolean")
    for optional_path in ("local_model_path", "adapter_path"):
        value = payload.get(optional_path)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{optional_path} must be a string or null")
    if not isinstance(payload["generation"], dict):
        raise ValueError("generation must be a JSON object")
    if not isinstance(payload["quantization"], dict):
        raise ValueError("quantization must be a JSON object")
    generation_payload = payload["generation"]
    generation_fields = {"seed", "do_sample", "temperature", "top_p", "max_tokens"}
    unknown_generation = sorted(set(generation_payload).difference(generation_fields))
    if unknown_generation:
        raise ValueError(
            f"unknown generation fields: {', '.join(unknown_generation)}"
        )
    integer_generation_fields = ("seed", "max_tokens")
    for field_name in integer_generation_fields:
        value = generation_payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"generation.{field_name} must be an integer")
    if not isinstance(generation_payload.get("do_sample"), bool):
        raise ValueError("generation.do_sample must be a boolean")
    for field_name in ("temperature", "top_p"):
        value = generation_payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"generation.{field_name} must be numeric")
    quantization_payload = payload["quantization"]
    quantization_fields = {"bits", "group_size", "mode"}
    unknown_quantization = sorted(
        set(quantization_payload).difference(quantization_fields)
    )
    if unknown_quantization:
        raise ValueError(
            f"unknown quantization fields: {', '.join(unknown_quantization)}"
        )
    for field_name in ("bits", "group_size"):
        value = quantization_payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"quantization.{field_name} must be an integer")
    if not isinstance(quantization_payload.get("mode"), str):
        raise ValueError("quantization.mode must be a string")
    generation = GenerationConfig(**generation_payload)
    quantization = QuantizationConfig(**quantization_payload)
    limits = payload.get("limits", {})
    if not isinstance(limits, dict):
        raise ValueError("limits must be a JSON object")
    unknown_limits = sorted(
        set(limits).difference({"max_image_pixels", "max_output_tokens"})
    )
    if unknown_limits:
        raise ValueError(f"unknown limit fields: {', '.join(unknown_limits)}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in limits.values()):
        raise ValueError("configured limits must be integers")
    if limits.get("max_image_pixels", 1) <= 0 or limits.get("max_output_tokens", 1) <= 0:
        raise ValueError("configured limits must be positive")
    return ModelConfig(
        schema_version=str(payload["schema_version"]),
        alias=str(payload["alias"]),
        family=str(payload["family"]),
        backend=str(payload["backend"]),
        model_id=str(payload["model_id"]),
        base_model_id=str(payload["base_model_id"]),
        revision=str(payload["revision"]),
        allow_download=payload["allow_download"],
        local_model_path=payload.get("local_model_path"),
        trust_remote_code=payload["trust_remote_code"],
        adapter_path=payload.get("adapter_path"),
        generation=generation,
        quantization=quantization,
        limits=limits,
    )
