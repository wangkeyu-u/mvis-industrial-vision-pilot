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
    model_mode: str = "auto"
    model_config_path: str = "configs/models/qwen3_vl_2b_mlx_4bit.json"
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
            "memory_budget_mb": self.memory_budget_mb,
            "min_disk_free_mb": self.min_disk_free_mb,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"service settings must be positive: {', '.join(invalid)}")
        if self.model_mode not in {"mock", "real", "auto"}:
            raise ValueError("model_mode must be mock, real, or auto")
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
        model = raw.get("model", {})
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
            model_mode=os.getenv(
                "MVIS_MODEL_MODE", str(model.get("mode", defaults.model_mode))
            ).lower(),
            model_config_path=os.getenv(
                "MVIS_MODEL_CONFIG",
                str(model.get("config_path", defaults.model_config_path)),
            ),
            # Administrative secrets are intentionally environment-only and are
            # never accepted from version-controlled YAML.
            modelops_token=os.getenv("MVIS_MODELOPS_TOKEN"),
            cors_allowed_origins=allowed_origins,
            memory_budget_mb=int(resources.get("memory_budget_mb", defaults.memory_budget_mb)),
            min_disk_free_mb=int(resources.get("min_disk_free_mb", defaults.min_disk_free_mb)),
            log_level=str(observability.get("log_level", defaults.log_level)),
        )
