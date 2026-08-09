"""Versionable service configuration with narrowly scoped environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    service_name: str = "multimodal-vision-service"
    service_version: str = "0.1.0"
    code_version: str = "unknown"
    api_schema_version: str = "1.0.0"
    config_path: str = "configs/service/default.yaml"
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    max_image_dimension: int = 8192
    inference_timeout_seconds: float = 8.0
    concurrency_limit: int = 1
    concurrency_wait_seconds: float = 0.1
    disconnect_poll_seconds: float = 0.05
    default_analysis_mode: str = "vlm_only"
    fusion_iou_threshold: float = 0.5
    model_mode: str = "auto"
    model_config_path: str = "configs/models/qwen3_vl_2b_mlx_4bit.json"
    model_weight_hash: str | None = None
    data_version: str = "ksdd-0.1.0"
    prompt_version: str = "ksdd_prompt_v1"
    active_model_alias: str = "zero_shot"
    zero_shot_quality_status: str = "pilot_failed"
    lora_model_id: str = "qwen3-vl-2b-instruct-4bit-lora-r8"
    lora_adapter_path: str | None = None
    lora_run_manifest_path: str | None = None
    lora_data_version: str = "ksdd_sft-1.0.0"
    lora_load_on_start: bool = True
    evaluator_attestation_path: str | None = None
    evaluator_signing_key: str | None = None
    specialist_id: str = "industrial-anomaly-specialist"
    specialist_manifest_path: str | None = None
    specialist_attestation_path: str | None = None
    specialist_quality_status: str = "unvalidated"
    specialist_data_version: str = "ksdd-0.1.0"
    specialist_prompt_version: str = "not-applicable"
    heatmap_max_items: int = 32
    heatmap_max_bytes: int = 4 * 1024 * 1024
    heatmap_ttl_seconds: int = 300
    heatmap_max_pixels: int = 4_194_304
    modelops_token: str | None = None
    cors_allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    )
    memory_budget_mb: int = 12 * 1024
    min_disk_free_mb: int = 2048
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        positive_values = {
            "max_image_bytes": self.max_image_bytes,
            "max_image_pixels": self.max_image_pixels,
            "max_image_dimension": self.max_image_dimension,
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "concurrency_limit": self.concurrency_limit,
            "concurrency_wait_seconds": self.concurrency_wait_seconds,
            "disconnect_poll_seconds": self.disconnect_poll_seconds,
            "heatmap_max_items": self.heatmap_max_items,
            "heatmap_max_bytes": self.heatmap_max_bytes,
            "heatmap_ttl_seconds": self.heatmap_ttl_seconds,
            "heatmap_max_pixels": self.heatmap_max_pixels,
            "memory_budget_mb": self.memory_budget_mb,
            "min_disk_free_mb": self.min_disk_free_mb,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"service settings must be positive: {', '.join(invalid)}")
        if self.model_mode not in {"mock", "real", "auto"}:
            raise ValueError("model_mode must be mock, real, or auto")
        if self.default_analysis_mode not in {"vlm_only", "specialist_only", "fused"}:
            raise ValueError("default_analysis_mode is invalid")
        if self.active_model_alias not in {"zero_shot", "lora"}:
            raise ValueError("active_model_alias must be zero_shot or lora")
        if not 0.0 <= self.fusion_iou_threshold <= 1.0:
            raise ValueError("fusion_iou_threshold must be in [0, 1]")
        if self.zero_shot_quality_status not in {
            "unvalidated",
            "pilot_failed",
            "pilot_candidate",
            "pilot_passed",
        }:
            raise ValueError("zero_shot_quality_status is invalid")
        if self.specialist_quality_status not in {
            "unvalidated",
            "pilot_failed",
            "pilot_candidate",
            "pilot_passed",
        }:
            raise ValueError("specialist_quality_status is invalid")
        for name, value in {
            "model_weight_hash": self.model_weight_hash,
        }.items():
            if value is not None and (
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if any(origin == "*" for origin in self.cors_allowed_origins):
            raise ValueError("wildcard CORS origins are not allowed")
        if self.modelops_token is not None and len(self.modelops_token) < 16:
            raise ValueError("MVIS_MODELOPS_TOKEN must contain at least 16 characters")

    @classmethod
    def load(cls, path: str | Path | None = None) -> ServiceSettings:
        defaults = cls()
        resolved = Path(path or os.getenv("MVIS_SERVICE_CONFIG", "configs/service/default.yaml"))
        raw: dict[str, Any] = {}
        if resolved.exists():
            with resolved.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

        service = raw.get("service", {})
        api = raw.get("api", {})
        image = raw.get("image", {})
        inference = raw.get("inference", {})
        fusion = raw.get("fusion", {})
        model = raw.get("model", {})
        specialist = raw.get("specialist", {})
        cors = raw.get("cors", {})
        resources = raw.get("resources", {})
        observability = raw.get("observability", {})
        configured_origins = cors.get("allowed_origins", list(defaults.cors_allowed_origins))
        if not isinstance(configured_origins, list) or not all(
            isinstance(origin, str) for origin in configured_origins
        ):
            raise ValueError("cors.allowed_origins must be a list of strings")
        env_origins = os.getenv("MVIS_CORS_ORIGINS")
        allowed_origins = (
            tuple(origin.strip() for origin in env_origins.split(",") if origin.strip())
            if env_origins is not None
            else tuple(configured_origins)
        )
        return cls(
            service_name=str(service.get("name", defaults.service_name)),
            service_version=str(service.get("version", defaults.service_version)),
            code_version=os.getenv(
                "MVIS_CODE_VERSION",
                str(service.get("code_version", defaults.code_version)),
            ),
            api_schema_version=str(api.get("schema_version", defaults.api_schema_version)),
            config_path=str(resolved),
            max_image_bytes=int(image.get("max_bytes", defaults.max_image_bytes)),
            max_image_pixels=int(image.get("max_pixels", defaults.max_image_pixels)),
            max_image_dimension=int(image.get("max_dimension", defaults.max_image_dimension)),
            inference_timeout_seconds=float(
                inference.get("timeout_seconds", defaults.inference_timeout_seconds)
            ),
            concurrency_limit=int(inference.get("concurrency_limit", defaults.concurrency_limit)),
            concurrency_wait_seconds=float(
                inference.get("concurrency_wait_seconds", defaults.concurrency_wait_seconds)
            ),
            disconnect_poll_seconds=float(
                inference.get("disconnect_poll_seconds", defaults.disconnect_poll_seconds)
            ),
            default_analysis_mode=str(
                os.getenv(
                    "MVIS_ANALYSIS_MODE",
                    str(fusion.get("default_mode", defaults.default_analysis_mode)),
                )
            ).lower(),
            fusion_iou_threshold=float(
                os.getenv(
                    "MVIS_FUSION_IOU_THRESHOLD",
                    str(fusion.get("iou_threshold", defaults.fusion_iou_threshold)),
                )
            ),
            model_mode=os.getenv(
                "MVIS_MODEL_MODE", str(model.get("mode", defaults.model_mode))
            ).lower(),
            model_config_path=os.getenv(
                "MVIS_MODEL_CONFIG",
                str(model.get("config_path", defaults.model_config_path)),
            ),
            model_weight_hash=(
                os.getenv("MVIS_MODEL_WEIGHT_HASH")
                or model.get("weight_hash")
                or defaults.model_weight_hash
            ),
            data_version=os.getenv(
                "MVIS_DATA_VERSION",
                str(model.get("data_version", defaults.data_version)),
            ),
            prompt_version=os.getenv(
                "MVIS_PROMPT_VERSION",
                str(model.get("prompt_version", defaults.prompt_version)),
            ),
            active_model_alias=os.getenv(
                "MVIS_ACTIVE_MODEL_ALIAS",
                str(model.get("active_alias", defaults.active_model_alias)),
            ).lower(),
            zero_shot_quality_status=str(
                model.get("zero_shot_quality_status", defaults.zero_shot_quality_status)
            ),
            lora_model_id=os.getenv(
                "MVIS_LORA_MODEL_ID",
                str(model.get("lora_model_id", defaults.lora_model_id)),
            ),
            lora_adapter_path=(
                os.getenv("MVIS_LORA_ADAPTER_PATH")
                or model.get("lora_adapter_path")
                or defaults.lora_adapter_path
            ),
            lora_run_manifest_path=(
                os.getenv("MVIS_LORA_RUN_MANIFEST")
                or model.get("lora_run_manifest_path")
                or defaults.lora_run_manifest_path
            ),
            lora_data_version=os.getenv(
                "MVIS_LORA_DATA_VERSION",
                str(model.get("lora_data_version", defaults.lora_data_version)),
            ),
            lora_load_on_start=(
                os.getenv("MVIS_LORA_LOAD_ON_START", "").strip().lower() in {"1", "true", "yes"}
                if os.getenv("MVIS_LORA_LOAD_ON_START") is not None
                else bool(model.get("lora_load_on_start", defaults.lora_load_on_start))
            ),
            evaluator_attestation_path=os.getenv("MVIS_EVALUATOR_ATTESTATION"),
            evaluator_signing_key=os.getenv("MVIS_EVALUATOR_SIGNING_KEY"),
            specialist_id=str(specialist.get("id", defaults.specialist_id)),
            specialist_manifest_path=(
                os.getenv("MVIS_SPECIALIST_MANIFEST")
                or specialist.get("manifest_path")
                or defaults.specialist_manifest_path
            ),
            specialist_attestation_path=os.getenv("MVIS_SPECIALIST_ATTESTATION"),
            specialist_quality_status=str(
                os.getenv(
                    "MVIS_SPECIALIST_QUALITY_STATUS",
                    str(
                        specialist.get(
                            "quality_status", defaults.specialist_quality_status
                        )
                    ),
                )
            ).lower(),
            specialist_data_version=str(
                specialist.get("data_version", defaults.specialist_data_version)
            ),
            specialist_prompt_version=str(
                specialist.get("prompt_version", defaults.specialist_prompt_version)
            ),
            heatmap_max_items=int(
                specialist.get("heatmap_max_items", defaults.heatmap_max_items)
            ),
            heatmap_max_bytes=int(
                specialist.get("heatmap_max_bytes", defaults.heatmap_max_bytes)
            ),
            heatmap_ttl_seconds=int(
                specialist.get("heatmap_ttl_seconds", defaults.heatmap_ttl_seconds)
            ),
            heatmap_max_pixels=int(
                specialist.get("heatmap_max_pixels", defaults.heatmap_max_pixels)
            ),
            # Administrative secrets are intentionally environment-only and are
            # never accepted from version-controlled YAML.
            modelops_token=os.getenv("MVIS_MODELOPS_TOKEN"),
            cors_allowed_origins=allowed_origins,
            memory_budget_mb=int(resources.get("memory_budget_mb", defaults.memory_budget_mb)),
            min_disk_free_mb=int(resources.get("min_disk_free_mb", defaults.min_disk_free_mb)),
            log_level=str(observability.get("log_level", defaults.log_level)),
        )
