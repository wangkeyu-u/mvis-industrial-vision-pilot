"""Build the service registry without coupling API startup to MLX imports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
from src.core.specialist_runtime import (
    SpecialistRegistration,
    UnavailableSpecialistAdapter,
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
SpecialistRegistrationLoader = Callable[[ServiceSettings], SpecialistRegistration]


def build_service_registry(
    settings: ServiceSettings,
    *,
    real_loader: RealRegistrationLoader | None = None,
    specialist_loader: SpecialistRegistrationLoader | None = None,
) -> ModelRegistry:
    """Select mock/real and make every fallback visible in registry status."""

    if settings.model_mode == "mock":
        registry = build_mock_registry()
        _attach_specialist(registry, settings, specialist_loader)
        return registry

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
        if settings.active_model_alias == "lora":
            if lora_registration.adapter.ready:
                registry.validate_candidate(
                    lora_registration.model_id,
                    actor="startup",
                    request_id="startup_lora_validate",
                    reason="explicit MVIS_ACTIVE_MODEL_ALIAS=lora",
                )
                registry.activate(
                    lora_registration.model_id,
                    actor="startup",
                    request_id="startup_lora_activate",
                    reason="explicit pilot candidate selection",
                    expected_active_model_id=real_registration.model_id,
                    expected_active_fingerprint=real_registration.model_fingerprint,
                    serving_tier=ServingTier.PILOT,
                )
            else:
                registry.set_runtime_status(
                    requested_mode=settings.model_mode,
                    selected_mode="real",
                    degraded=True,
                    fallback_reason="explicit_lora_candidate_not_ready",
                )
    if registry.runtime_status()["requested_mode"] == "unconfigured":
        registry.set_runtime_status(
            requested_mode=settings.model_mode,
            selected_mode="real",
            degraded=False,
        )
    _attach_specialist(registry, settings, specialist_loader)
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
    _attach_specialist(registry, settings, None)
    return registry


def _attach_specialist(
    registry: ModelRegistry,
    settings: ServiceSettings,
    loader: SpecialistRegistrationLoader | None,
) -> None:
    try:
        registration = loader(settings) if loader is not None else _build_specialist_registration(settings)
    except Exception:  # noqa: BLE001 - specialist mode must fail explicitly, not service startup
        registration = _unavailable_specialist_registration(
            settings,
            "specialist_initialization_failed",
        )
    registry.register_specialist(registration)


def _build_specialist_registration(settings: ServiceSettings) -> SpecialistRegistration:
    """Load a backend bridge when published; otherwise preserve a named unavailable entry."""

    manifest = _read_specialist_manifest(settings.specialist_manifest_path)
    if manifest is None:
        return _unavailable_specialist_registration(
            settings,
            "specialist_artifact_unavailable",
        )
    try:
        from src.inference.service_bridge import create_specialist_service_adapter
    except (ImportError, AttributeError):
        return _unavailable_specialist_registration(
            settings,
            "specialist_bridge_unavailable",
            manifest=manifest,
        )
    try:
        adapter = create_specialist_service_adapter(settings.specialist_manifest_path)
    except Exception as exc:
        raise RealAdapterUnavailable("specialist_load_failed") from exc
    if not adapter.ready:
        return _unavailable_specialist_registration(
            settings,
            "specialist_load_incomplete",
            manifest=manifest,
        )
    provenance = _specialist_provenance(settings, manifest, adapter.identity)
    status, evidence = _specialist_quality_for_provenance(settings, provenance)
    return SpecialistRegistration(
        specialist_id=settings.specialist_id,
        adapter=adapter,
        source=f"specialist-manifest:{Path(settings.specialist_manifest_path).name}",
        provenance=provenance,
        quality_status=status,
        quality_evidence=evidence,
        serving_tier=ServingTier.PILOT,
    )


def _unavailable_specialist_registration(
    settings: ServiceSettings,
    reason_code: str,
    *,
    manifest: dict[str, object] | None = None,
) -> SpecialistRegistration:
    identity = _specialist_identity(settings, manifest)
    provenance = _specialist_provenance(settings, manifest, identity)
    status, evidence = _specialist_quality_for_provenance(settings, provenance)
    return SpecialistRegistration(
        specialist_id=settings.specialist_id,
        adapter=UnavailableSpecialistAdapter(identity, reason_code),
        source=(
            f"specialist-manifest:{Path(settings.specialist_manifest_path).name}"
            if settings.specialist_manifest_path
            else "specialist-manifest:awaiting-artifact"
        ),
        provenance=provenance,
        quality_status=status,
        quality_evidence=evidence,
        serving_tier=ServingTier.PILOT,
    )


def _read_specialist_manifest(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size <= 0
            or candidate.stat().st_size > 1024 * 1024
        ):
            return None
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _specialist_identity(
    settings: ServiceSettings,
    manifest: dict[str, object] | None,
) -> ModelIdentity:
    details = _manifest_identity_details(manifest)
    return ModelIdentity(
        base=str(details.get("model_id") or settings.specialist_id),
        revision=str(details.get("revision") or "artifact-unavailable"),
    )


def _specialist_provenance(
    settings: ServiceSettings,
    manifest: dict[str, object] | None,
    identity: ModelIdentity,
) -> ModelProvenance:
    details = _manifest_identity_details(manifest)
    manifest_hash = _file_fingerprint(settings.specialist_manifest_path) if manifest else None
    config_fingerprint = str(details.get("config_fingerprint") or manifest_hash or "0" * 64)
    if len(config_fingerprint) not in {16, 64}:
        config_fingerprint = hashlib.sha256(config_fingerprint.encode()).hexdigest()
    artifact_hash = details.get("artifact_sha256") or details.get("weight_sha256")
    if not isinstance(artifact_hash, str) or len(artifact_hash) != 64:
        artifact_hash = None
    return ModelProvenance(
        model_id=identity.base,
        checkpoint_revision=identity.revision,
        backend=str(details.get("backend") or "anomaly-specialist"),
        config_fingerprint=config_fingerprint,
        config_artifact_sha256=manifest_hash,
        adapter_hash=None,
        weight_hash=artifact_hash,
        data_version=str(details.get("data_version") or settings.specialist_data_version),
        prompt_version=settings.specialist_prompt_version,
    )


def _manifest_identity_details(manifest: dict[str, object] | None) -> dict[str, object]:
    if manifest is None:
        return {}
    for key in ("specialist", "model", "identity"):
        value = manifest.get(key)
        if isinstance(value, dict):
            return value
    configuration = manifest.get("configuration")
    artifacts = manifest.get("artifacts")
    if isinstance(configuration, dict) and isinstance(artifacts, dict):
        algorithm = str(configuration.get("algorithm") or "specialist")
        backbone = str(configuration.get("backbone") or "unknown")
        return {
            "model_id": f"anomalib/{algorithm}-{backbone}-ksdd-v0",
            "revision": artifacts.get("checkpoint_sha256"),
            "artifact_sha256": artifacts.get("checkpoint_sha256"),
            "backend": f"anomalib-{algorithm}",
            "data_version": configuration.get("dataset_version"),
        }
    return manifest


def _specialist_quality_for_provenance(
    settings: ServiceSettings,
    provenance: ModelProvenance,
) -> tuple[QualityStatus, QualityEvidence]:
    verified, evidence = verify_evaluator_attestation(
        settings.specialist_attestation_path,
        settings.evaluator_signing_key,
        provenance,
    )
    if verified is QualityStatus.PILOT_PASSED:
        return verified, evidence
    configured = QualityStatus(settings.specialist_quality_status)
    if configured is QualityStatus.PILOT_PASSED:
        configured = QualityStatus.PILOT_CANDIDATE
    return configured, evidence


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
    manifest_failure = _lora_manifest_failure(settings, base, adapter_hash)
    if manifest_failure is not None:
        adapter = UnavailableModelAdapter(identity, manifest_failure)
        reason = manifest_failure
    elif adapter_hash is not None and settings.lora_load_on_start:
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
        quantization="4-bit+lora",
        weight_hash=base.weight_hash,
        config_fingerprint=config_fingerprint,
        provenance=provenance,
        quality_status=quality_status,
        quality_evidence=quality_evidence,
    )


def _lora_manifest_failure(
    settings: ServiceSettings,
    base: ModelProvenance,
    adapter_hash: str | None,
) -> str | None:
    if settings.lora_run_manifest_path is None:
        return None
    manifest = _read_specialist_manifest(settings.lora_run_manifest_path)
    if manifest is None:
        return "lora_run_manifest_invalid"
    artifacts = manifest.get("artifacts")
    model = manifest.get("model")
    if not isinstance(artifacts, dict) or not isinstance(model, dict):
        return "lora_run_manifest_contract_invalid"
    if manifest.get("status") != "completed" or manifest.get("run_kind") != "formal":
        return "lora_run_not_completed"
    if artifacts.get("adapter_sha256") != adapter_hash:
        return "lora_adapter_hash_mismatch"
    if model.get("revision") != base.checkpoint_revision:
        return "lora_base_revision_mismatch"
    if model.get("weight_sha256") != base.weight_hash:
        return "lora_base_weight_hash_mismatch"
    return None


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
    canonical_adapter = candidate / "adapters.safetensors"
    if canonical_adapter in regular_files:
        return _stream_sha256(canonical_adapter)
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
