"""Build the service registry without coupling API startup to MLX imports."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path

from src.core.config import ServiceSettings
from src.core.errors import ErrorCode, ServiceError
from src.core.model_registry import (
    AdapterRequest,
    ModelRegistration,
    ModelRegistry,
    ModelState,
    build_mock_registry,
)
from src.core.schemas import ModelIdentity, ModelOutput


class RealAdapterUnavailable(RuntimeError):
    """A safe, machine-readable reason why the real adapter is unavailable."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class UnavailableModelAdapter:
    def __init__(self, identity: ModelIdentity, reason_code: str) -> None:
        self._identity = identity
        self.reason_code = reason_code

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def ready(self) -> bool:
        return False

    async def analyze(self, request: AdapterRequest) -> ModelOutput | Mapping:
        raise ServiceError(
            ErrorCode.MODEL_NOT_READY,
            f"real model is not ready ({self.reason_code})",
        )


RealRegistrationLoader = Callable[[ServiceSettings], ModelRegistration]


def build_service_registry(
    settings: ServiceSettings,
    *,
    real_loader: RealRegistrationLoader | None = None,
) -> ModelRegistry:
    """Select mock/real and make every fallback visible in registry status."""

    if settings.model_mode == "mock":
        return build_mock_registry()

    loader = real_loader or _build_real_registration
    try:
        real_registration = loader(settings)
    except RealAdapterUnavailable as exc:
        return _unavailable_registry(settings, exc.reason_code)
    except Exception as exc:  # noqa: BLE001 - startup must degrade deterministically
        # Do not expose local paths or dependency exception messages via health APIs.
        reason = (
            "model_config_invalid"
            if isinstance(exc, (OSError, ValueError))
            else "real_adapter_initialization_failed"
        )
        return _unavailable_registry(settings, reason)

    registry = ModelRegistry()
    registry.register(real_registration)
    registry.set_runtime_status(
        requested_mode=settings.model_mode,
        selected_mode="real",
        degraded=False,
    )
    return registry


def _build_real_registration(settings: ServiceSettings) -> ModelRegistration:
    config_fingerprint = _file_fingerprint(settings.model_config_path)
    if config_fingerprint is None:
        raise RealAdapterUnavailable("model_config_unavailable")
    if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_vlm") is None:
        raise RealAdapterUnavailable("mlx_runtime_unavailable")

    try:
        from src.inference.factory import create_adapter
        from src.inference.service_bridge import AsyncServiceAdapterBridge

        adapter = create_adapter(settings.model_config_path)
        adapter.load()
    except (ImportError, ModuleNotFoundError) as exc:
        raise RealAdapterUnavailable("mlx_runtime_unavailable") from exc
    except Exception as exc:
        raise RealAdapterUnavailable("model_load_failed") from exc

    bridge = AsyncServiceAdapterBridge(adapter)
    if not bridge.ready:
        raise RealAdapterUnavailable("model_load_incomplete")
    return ModelRegistration(
        model_id=adapter.config.alias,
        adapter=bridge,
        state=ModelState.ACTIVE,
        source=f"model-config:{Path(settings.model_config_path).name}",
        quantization=f"{adapter.config.quantization.bits}-bit",
        weight_hash=None,
        config_fingerprint=config_fingerprint,
    )


def _unavailable_registry(settings: ServiceSettings, reason_code: str) -> ModelRegistry:
    identity, model_id, quantization = _configured_identity(settings)
    placeholder = ModelRegistration(
        model_id=model_id,
        adapter=UnavailableModelAdapter(identity, reason_code),
        state=(ModelState.ACTIVE if settings.model_mode == "real" else ModelState.CANDIDATE),
        source=f"model-config:{Path(settings.model_config_path).name}",
        quantization=quantization,
        weight_hash=None,
        config_fingerprint=_file_fingerprint(settings.model_config_path),
    )

    if settings.model_mode == "real":
        registry = ModelRegistry()
        registry.register(placeholder)
        selected_mode = "real"
    else:
        registry = build_mock_registry()
        registry.register(placeholder)
        selected_mode = "mock"
    registry.set_runtime_status(
        requested_mode=settings.model_mode,
        selected_mode=selected_mode,
        degraded=True,
        fallback_reason=reason_code,
    )
    return registry


def _configured_identity(
    settings: ServiceSettings,
) -> tuple[ModelIdentity, str, str | None]:
    try:
        from src.inference.config import load_model_config

        config = load_model_config(settings.model_config_path)
    except (ImportError, OSError, ValueError):
        return ModelIdentity(base="unavailable-real-model"), "real-model", None
    return (
        ModelIdentity(
            base=config.alias,
            adapter=Path(config.adapter_path).name if config.adapter_path else None,
        ),
        config.alias,
        f"{config.quantization.bits}-bit",
    )


def _file_fingerprint(path: str | Path) -> str | None:
    """Hash a bounded regular config file without exposing its local path."""

    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size > 1024 * 1024:
            return None
        return hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        return None
