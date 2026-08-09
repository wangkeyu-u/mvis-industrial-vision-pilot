"""Build the service registry without coupling API startup to MLX imports."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from src.core.config import ServiceSettings
from src.core.errors import ErrorCode, ServiceError
from src.core.model_registry import (
    AdapterRequest,
    ModelRegistration,
    ModelRegistry,
    ModelState,
    build_mock_registry,
    fingerprint_mapping,
)
from src.core.quality_gate import verify_evaluator_attestation
from src.core.schemas import (
    ModelIdentity,
    ModelOutput,
    ModelProvenance,
    QualityEvidence,
    QualityStatus,
    ServingTier,
)


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
        return _unavailable_registry(
            settings,
            exc.reason_code,
            include_lora=real_loader is None,
        )
    except Exception as exc:  # noqa: BLE001 - startup must degrade deterministically
        # Do not expose local paths or dependency exception messages via health APIs.
        reason = (
            "model_config_invalid"
            if isinstance(exc, (OSError, ValueError))
            else "real_adapter_initialization_failed"
        )
        return _unavailable_registry(settings, reason, include_lora=real_loader is None)

    registry = ModelRegistry()
    registry.register(real_registration)
    if real_loader is None:
        lora_registration = _build_lora_registration(settings, real_registration.provenance)
        registry.register(lora_registration)
        registry.set_alias("zero_shot", real_registration.model_id)
        registry.set_alias("lora", lora_registration.model_id)
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
    provenance = ModelProvenance(
        model_id=adapter.config.model_id,
        checkpoint_revision=adapter.config.revision,
        backend=adapter.config.backend,
        config_fingerprint=adapter.config.fingerprint,
        config_artifact_sha256=config_fingerprint,
        adapter_hash=None,
        weight_hash=settings.model_weight_hash,
        data_version=settings.data_version,
        prompt_version=settings.prompt_version,
    )
    quality_status, quality_evidence = _quality_for_provenance(
        settings,
        provenance,
        configured=QualityStatus(settings.zero_shot_quality_status),
    )
    return ModelRegistration(
        model_id=adapter.config.alias,
        adapter=bridge,
        state=ModelState.ACTIVE,
        source=f"model-config:{Path(settings.model_config_path).name}",
        quantization=f"{adapter.config.quantization.bits}-bit",
        weight_hash=settings.model_weight_hash,
        config_fingerprint=config_fingerprint,
        provenance=provenance,
        quality_status=quality_status,
        quality_evidence=quality_evidence,
        serving_tier=ServingTier.PILOT,
    )


def _unavailable_registry(
    settings: ServiceSettings,
    reason_code: str,
    *,
    include_lora: bool,
) -> ModelRegistry:
    identity, model_id, quantization = _configured_identity(settings)
    provenance = _configured_provenance(settings)
    quality_status, quality_evidence = _quality_for_provenance(
        settings,
        provenance,
        configured=QualityStatus(settings.zero_shot_quality_status),
    )
    placeholder = ModelRegistration(
        model_id=model_id,
        adapter=UnavailableModelAdapter(identity, reason_code),
        state=(ModelState.ACTIVE if settings.model_mode == "real" else ModelState.CANDIDATE),
        source=f"model-config:{Path(settings.model_config_path).name}",
        quantization=quantization,
        weight_hash=settings.model_weight_hash,
        config_fingerprint=_file_fingerprint(settings.model_config_path),
        provenance=provenance,
        quality_status=quality_status,
        quality_evidence=quality_evidence,
        serving_tier=(ServingTier.PILOT if settings.model_mode == "real" else None),
    )

    if settings.model_mode == "real":
        registry = ModelRegistry()
        registry.register(placeholder)
        selected_mode = "real"
    else:
        registry = build_mock_registry()
        registry.register(placeholder)
        selected_mode = "mock"
    if include_lora:
        lora_registration = _build_lora_registration(settings, provenance)
        registry.register(lora_registration)
        registry.set_alias("zero_shot", placeholder.model_id)
        registry.set_alias("lora", lora_registration.model_id)
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
            revision=config.revision,
        ),
        config.alias,
        f"{config.quantization.bits}-bit",
    )


def _configured_provenance(settings: ServiceSettings) -> ModelProvenance:
    config_fingerprint = _file_fingerprint(settings.model_config_path) or "0" * 64
    try:
        from src.inference.config import load_model_config

        config = load_model_config(settings.model_config_path)
        return ModelProvenance(
            model_id=config.model_id,
            checkpoint_revision=config.revision,
            backend=config.backend,
            config_fingerprint=config.fingerprint,
            config_artifact_sha256=config_fingerprint,
            adapter_hash=None,
            weight_hash=settings.model_weight_hash,
            data_version=settings.data_version,
            prompt_version=settings.prompt_version,
        )
    except (ImportError, OSError, ValueError):
        return ModelProvenance(
            model_id="unavailable-real-model",
            checkpoint_revision="config-unavailable",
            backend="mlx_vlm",
            config_fingerprint=config_fingerprint,
            config_artifact_sha256=(config_fingerprint if config_fingerprint != "0" * 64 else None),
            adapter_hash=None,
            weight_hash=settings.model_weight_hash,
            data_version=settings.data_version,
            prompt_version=settings.prompt_version,
        )


def _build_lora_registration(
    settings: ServiceSettings,
    base_provenance: ModelProvenance | None,
) -> ModelRegistration:
    base = base_provenance or _configured_provenance(settings)
    adapter_hash = _artifact_fingerprint(settings.lora_adapter_path)
    config_fingerprint = fingerprint_mapping(
        {
            "base_config": base.config_fingerprint,
            "adapter_hash": adapter_hash,
            "model_id": settings.lora_model_id,
            "data_version": settings.lora_data_version,
            "prompt_version": settings.prompt_version,
        }
    )
    provenance = ModelProvenance(
        model_id=base.model_id,
        checkpoint_revision=base.checkpoint_revision,
        backend=base.backend,
        config_fingerprint=_lora_algorithm_config_fingerprint(settings),
        config_artifact_sha256=base.config_artifact_sha256,
        adapter_hash=adapter_hash,
        weight_hash=base.weight_hash,
        data_version=settings.lora_data_version,
        prompt_version=settings.prompt_version,
    )
    identity = ModelIdentity(
        base=settings.lora_model_id,
        adapter=(Path(settings.lora_adapter_path).name if settings.lora_adapter_path else None),
        revision=base.checkpoint_revision,
    )
    adapter: object
    reason = "lora_artifact_unavailable"
    if adapter_hash is not None and settings.lora_load_on_start:
        try:
            from src.inference.config import load_model_config
            from src.inference.mlx_backend import MlxVlmBackend
            from src.inference.qwen3_vl import Qwen3VLAdapter
            from src.inference.service_bridge import AsyncServiceAdapterBridge

            config = replace(
                load_model_config(settings.model_config_path),
                alias=settings.lora_model_id,
                adapter_path=settings.lora_adapter_path,
            )
            model_adapter = Qwen3VLAdapter(config=config, backend=MlxVlmBackend(config))
            model_adapter.load()
            adapter = AsyncServiceAdapterBridge(model_adapter)
            reason = ""
        except Exception:  # noqa: BLE001 - candidate remains safely unavailable
            adapter = UnavailableModelAdapter(identity, "lora_load_failed")
            reason = "lora_load_failed"
    else:
        if adapter_hash is not None:
            reason = "lora_load_disabled"
        adapter = UnavailableModelAdapter(identity, reason)

    quality_status, quality_evidence = _quality_for_provenance(
        settings,
        provenance,
        configured=(
            QualityStatus.PILOT_CANDIDATE
            if getattr(adapter, "ready", False)
            else QualityStatus.UNVALIDATED
        ),
    )
    return ModelRegistration(
        model_id=settings.lora_model_id,
        adapter=adapter,  # type: ignore[arg-type]
        state=ModelState.CANDIDATE,
        source=(
            f"lora-adapter:{Path(settings.lora_adapter_path).name}"
            if settings.lora_adapter_path
            else "lora-adapter:awaiting-artifact"
        ),
        quantization="4-bit+lora-r8",
        weight_hash=base.weight_hash,
        config_fingerprint=config_fingerprint,
        provenance=provenance,
        quality_status=quality_status,
        quality_evidence=quality_evidence,
    )


def _quality_for_provenance(
    settings: ServiceSettings,
    provenance: ModelProvenance,
    *,
    configured: QualityStatus,
) -> tuple[QualityStatus, QualityEvidence]:
    verified_status, evidence = verify_evaluator_attestation(
        settings.evaluator_attestation_path,
        settings.evaluator_signing_key,
        provenance,
    )
    if verified_status is QualityStatus.PILOT_PASSED:
        return verified_status, evidence
    # Merely configuring pilot_passed cannot bypass the detached signature gate.
    if configured is QualityStatus.PILOT_PASSED:
        configured = QualityStatus.PILOT_CANDIDATE
    return configured, evidence


def _lora_algorithm_config_fingerprint(settings: ServiceSettings) -> str:
    try:
        from src.inference.config import load_model_config

        config = replace(
            load_model_config(settings.model_config_path),
            alias=settings.lora_model_id,
            adapter_path=settings.lora_adapter_path,
        )
        return config.fingerprint
    except (ImportError, OSError, ValueError):
        return fingerprint_mapping(
            {
                "model_id": settings.lora_model_id,
                "adapter_path_configured": settings.lora_adapter_path is not None,
                "data_version": settings.lora_data_version,
                "prompt_version": settings.prompt_version,
            }
        )


def _artifact_fingerprint(path: str | Path | None) -> str | None:
    """Hash adapter weights (or a deterministic tree fallback) without symlinks."""

    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or candidate.is_symlink():
        return None
    entries = [candidate] if candidate.is_file() else sorted(candidate.rglob("*"))
    if any(item.is_symlink() for item in entries):
        return None
    regular_files = [item for item in entries if item.is_file()]
    if not regular_files:
        return None
    safetensors = [item for item in regular_files if item.suffix == ".safetensors"]
    if candidate.is_file():
        return _stream_sha256(candidate)
    if len(safetensors) == 1:
        # mlx-vlm emits one adapter weights file plus a small JSON config. The
        # standard adapter hash is the weights-file digest reported by training.
        return _stream_sha256(safetensors[0])

    digest = hashlib.sha256()
    try:
        for item in regular_files:
            relative = str(item.relative_to(candidate))
            digest.update(relative.encode("utf-8"))
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _stream_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _file_fingerprint(path: str | Path) -> str | None:
    """Hash a bounded regular config file without exposing its local path."""

    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size > 1024 * 1024:
            return None
        return hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        return None
