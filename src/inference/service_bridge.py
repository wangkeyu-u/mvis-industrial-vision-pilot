"""Thin compatibility bridge to the existing async service-side protocol.

The framework-neutral adapter remains the source of model behavior.  This
module only translates existing ``src.core`` request/result objects and keeps
the blocking MLX call off the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from PIL import Image

from src.core.errors import ErrorCode, ServiceError
from src.core.model_registry import AdapterRequest
from src.core.schemas import (
    EvidenceObject,
    ModelIdentity,
    ModelOutput,
    ObjectSource,
    SpecialistOutput,
)
from src.core.specialist_runtime import SpecialistRequest

from .base import GenerationBackend
from .config import load_model_config
from .contracts import GenerationConfig, ModelRequest, Task
from .errors import ModelNotReadyError, ModelOutputError
from .mlx_backend import MlxVlmBackend
from .qwen3_vl import Qwen3VLAdapter


class PatchcoreServiceAdapterBridge:
    """Loaded, hash-pinned PatchCore runtime implementing SpecialistAdapter."""

    def __init__(
        self,
        *,
        model: Any,
        torch_module: Any,
        device: Any,
        image_size: int,
        image_threshold: float,
        bbox_threshold: float,
        checkpoint_sha256: str,
    ) -> None:
        self._model = model
        self._torch = torch_module
        self._device = device
        self._image_size = image_size
        self._image_threshold = image_threshold
        self._bbox_threshold = bbox_threshold
        self._checkpoint_sha256 = checkpoint_sha256
        self._lock = threading.Lock()

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            base="anomalib/patchcore-resnet18-ksdd-v0",
            revision=self._checkpoint_sha256,
        )

    @property
    def ready(self) -> bool:
        return self._model is not None

    async def detect(self, request: SpecialistRequest) -> SpecialistOutput | Mapping:
        # Cancellation is intentionally not caught. asyncio.to_thread propagates
        # CancelledError immediately while the short native inference unwinds.
        return await asyncio.to_thread(self._detect_sync, request)

    def _detect_sync(self, request: SpecialistRequest) -> SpecialistOutput:
        import numpy as np

        from .patchcore_specialist import heatmap_to_bbox, normalize_heatmap

        try:
            with Image.open(io.BytesIO(request.image_bytes)) as image:
                rgb = image.convert("RGB")
                if rgb.size != (request.image.width, request.image.height):
                    raise ValueError("decoded specialist image dimensions differ from metadata")
                original_width, original_height = rgb.size
                resized = rgb.resize(
                    (self._image_size, self._image_size), Image.Resampling.BILINEAR
                )
                values = np.asarray(resized, dtype=np.float32) / 255.0
        except (OSError, ValueError) as exc:
            raise ServiceError(
                ErrorCode.INVALID_IMAGE, "specialist image decoding failed"
            ) from exc
        tensor = self._torch.from_numpy(values).permute(2, 0, 1)
        mean = self._torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = self._torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self._device)
        with self._lock, self._torch.inference_mode():
            prediction = self._model(tensor)
            if self._device.type == "mps":
                self._torch.mps.synchronize()
        score = max(0.0, float(prediction.pred_score.detach().cpu().reshape(-1)[0]))
        heatmap = normalize_heatmap(prediction.anomaly_map.detach().cpu().numpy().squeeze())
        positive = score >= self._image_threshold
        bbox = (
            heatmap_to_bbox(
                heatmap,
                self._bbox_threshold,
                original_width,
                original_height,
            )
            if positive
            else None
        )
        confidence = min(
            0.999,
            0.5
            + abs(score - self._image_threshold)
            / (2.0 * max(self._image_threshold, 1e-6)),
        )
        objects = (
            [
                EvidenceObject(
                    label="surface_defect",
                    bbox=bbox,
                    confidence=confidence,
                    source=ObjectSource.PATCHCORE,
                )
            ]
            if bbox is not None
            else []
        )
        png_image = Image.fromarray(np.uint8(np.clip(heatmap * 255, 0, 255)), mode="L")
        png_image = png_image.resize(
            (original_width, original_height), Image.Resampling.BILINEAR
        )
        png_buffer = io.BytesIO()
        png_image.save(png_buffer, format="PNG")
        return SpecialistOutput(
            score=score,
            threshold=self._image_threshold,
            source=ObjectSource.PATCHCORE,
            objects=objects,
            heatmap_png=png_buffer.getvalue(),
            warnings=["patchcore_validation_calibrated"],
        )


class UnetServiceAdapterBridge:
    """Loaded, hash-pinned phase 8 U-Net segmentation specialist adapter."""

    def __init__(
        self,
        *,
        model: Any,
        torch_module: Any,
        device: Any,
        input_size: int,
        tile_size: int,
        tile_stride: int,
        image_threshold: float,
        fusion_mode: str,
        postprocess_config: Any,
        checkpoint_sha256: str,
    ) -> None:
        self._model = model
        self._torch = torch_module
        self._device = device
        self._input_size = input_size
        self._tile_size = tile_size
        self._tile_stride = tile_stride
        self._image_threshold = image_threshold
        self._fusion_mode = fusion_mode
        self._postprocess_config = postprocess_config
        self._checkpoint_sha256 = checkpoint_sha256
        self._lock = threading.Lock()

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            base="mvis/unet-resnet18-ksdd-v1",
            revision=self._checkpoint_sha256,
        )

    @property
    def ready(self) -> bool:
        return self._model is not None

    async def detect(self, request: SpecialistRequest) -> SpecialistOutput | Mapping:
        return await asyncio.to_thread(self._detect_sync, request)

    def _detect_sync(self, request: SpecialistRequest) -> SpecialistOutput:
        import numpy as np

        from src.evaluation.localization_postprocess import (
            extract_boxes,
            scale_box_to_original,
        )
        from src.training.unet_segmentation import _fuse_grid, _grid_offsets

        try:
            with Image.open(io.BytesIO(request.image_bytes)) as image:
                rgb = image.convert("RGB")
                if rgb.size != (request.image.width, request.image.height):
                    raise ValueError("decoded specialist image dimensions differ from metadata")
                original_width, original_height = rgb.size
        except (OSError, ValueError) as exc:
            raise ServiceError(
                ErrorCode.INVALID_IMAGE, "specialist image decoding failed"
            ) from exc

        torch = self._torch
        x_offsets = _grid_offsets(original_width, self._tile_size, self._tile_stride)
        y_offsets = _grid_offsets(original_height, self._tile_size, self._tile_stride)
        tiles = []
        positions = []
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with self._lock, torch.inference_mode():
            for y_offset in y_offsets:
                for x_offset in x_offsets:
                    crop = rgb.crop(
                        (x_offset, y_offset, x_offset + self._tile_size, y_offset + self._tile_size)
                    )
                    if self._tile_size != self._input_size:
                        crop = crop.resize(
                            (self._input_size, self._input_size), Image.Resampling.BILINEAR
                        )
                    array = np.asarray(crop, dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
                    batch = ((tensor[None] - mean) / std).to(self._device)
                    probs = torch.sigmoid(self._model(batch)).detach().cpu().numpy()[0, 0]
                    tiles.append(probs.astype(np.float32))
                    positions.append((x_offset, y_offset))
            if self._device.type == "mps":
                torch.mps.synchronize()
        grid_item = {
            "width": original_width,
            "height": original_height,
            "tiles": tiles,
            "positions": positions,
        }
        from types import SimpleNamespace

        fused = _fuse_grid(
            grid_item, SimpleNamespace(tile_size=self._tile_size), self._fusion_mode
        )
        score = float(fused.max()) if fused.size else 0.0
        positive = score >= self._image_threshold
        boxes, binary, _ = extract_boxes(fused, self._postprocess_config)
        scaled_boxes = [
            scale_box_to_original(
                box, binary.shape, (original_width, original_height)
            )
            for box in boxes
        ]
        int_boxes = [
            (
                max(0, min(original_width - 1, int(math.floor(box[0])))),
                max(0, min(original_height - 1, int(math.floor(box[1])))),
                max(1, min(original_width, int(math.ceil(box[2])))),
                max(1, min(original_height, int(math.ceil(box[3])))),
            )
            for box in scaled_boxes
        ]
        confidence = min(
            0.999,
            0.5
            + abs(score - self._image_threshold)
            / (2.0 * max(self._image_threshold, 1e-6)),
        )
        objects = (
            [
                EvidenceObject(
                    label="surface_defect",
                    bbox=box,
                    confidence=confidence,
                    source=ObjectSource.UNET,
                )
                for box in int_boxes
            ]
            if positive
            else []
        )
        png_image = Image.fromarray(
            np.uint8(np.clip(fused / max(float(fused.max()), 1e-6) * 255, 0, 255)), mode="L"
        )
        png_buffer = io.BytesIO()
        png_image.save(png_buffer, format="PNG")
        return SpecialistOutput(
            score=score,
            threshold=self._image_threshold,
            source=ObjectSource.UNET,
            objects=objects,
            heatmap_png=png_buffer.getvalue(),
            warnings=["phase8_unet_validation_calibrated"],
        )


class AsyncServiceAdapterBridge:
    def __init__(self, adapter: Qwen3VLAdapter) -> None:
        self.adapter = adapter

    @property
    def identity(self) -> ModelIdentity:
        adapter_name = (
            Path(self.adapter.config.adapter_path).name
            if self.adapter.config.adapter_path
            else None
        )
        return ModelIdentity(
            base=self.adapter.config.alias,
            adapter=adapter_name,
        )

    @property
    def ready(self) -> bool:
        return self.adapter.ready

    def load(self) -> None:
        """Load the underlying adapter before registering it as active."""

        self.adapter.load()

    async def analyze(self, request: AdapterRequest) -> ModelOutput | Mapping[str, object]:
        try:
            return await asyncio.to_thread(self._analyze_sync, request)
        except ModelOutputError:
            # The established AnalyzeService maps Pydantic validation failures
            # after the adapter call to OUTPUT_VALIDATION_FAILED. Returning an
            # intentionally invalid provider-neutral mapping uses that stable
            # path without exposing raw model text.
            return {"adapter_output_invalid": True}
        except ModelNotReadyError as exc:
            raise ServiceError(
                ErrorCode.MODEL_NOT_READY,
                "configured model runtime is not ready",
            ) from exc
        except ValueError as exc:
            raise ServiceError(
                ErrorCode.INVALID_QUERY,
                "model options exceed the registered configuration",
            ) from exc

    def _analyze_sync(self, request: AdapterRequest) -> ModelOutput:
        with self._backend_image(request) as image:
            generation = GenerationConfig(
                seed=request.options.seed,
                do_sample=request.options.temperature > 0.0,
                temperature=request.options.temperature,
                top_p=request.options.top_p,
                max_tokens=request.options.max_tokens,
            )
            result = self.adapter.analyze(
                ModelRequest(
                    image=image,
                    image_width=request.image.width,
                    image_height=request.image.height,
                    query=request.query,
                    task=Task(request.task.value),
                    use_specialist=request.use_specialist,
                    generation=generation,
                )
            )
        warnings = list(result.warnings)
        if result.refusal is not None:
            # API schema 1.0 has no refusal field. Preserve the machine-readable
            # code without changing the established public schema.
            warnings.append(f"refusal:{result.refusal.code.value}")
        return ModelOutput(
            result=result.decision.value,
            objects=[
                EvidenceObject(
                    label=item.label,
                    bbox=item.bbox,
                    confidence=item.confidence,
                    source=ObjectSource.VLM,
                )
                for item in result.objects
            ],
            reason=result.reason,
            uncertain=result.uncertain,
            warnings=warnings,
        )

    @contextmanager
    def _backend_image(self, request: AdapterRequest) -> Iterator[bytes | str]:
        if self.adapter.backend.name != "mlx_vlm":
            yield request.image_bytes
            return
        suffix = ".jpg" if request.image.format == "jpeg" else f".{request.image.format}"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(request.image_bytes)
                temp_path = Path(handle.name)
            yield str(temp_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def create_service_adapter(
    config_path: str | Path,
    *,
    backend: GenerationBackend | None = None,
    load: bool = False,
) -> AsyncServiceAdapterBridge:
    """Create an object directly compatible with ``core.ModelRegistry``.

    Supplying ``MockBackend`` keeps construction dependency-free. With no
    override, the MLX implementation remains lazy and never downloads while
    the registered configuration has ``allow_download=false``.
    """

    config = load_model_config(config_path)
    runtime = backend if backend is not None else MlxVlmBackend(config)
    bridge = AsyncServiceAdapterBridge(Qwen3VLAdapter(config, runtime))
    if load:
        bridge.load()
    return bridge


def create_specialist_service_adapter(
    manifest_path: str | Path,
) -> PatchcoreServiceAdapterBridge | UnetServiceAdapterBridge:
    """Load one offline specialist checkpoint after validating its run evidence.

    Phase 7 PatchCore manifests keep the original loading path; phase 8 U-Net
    specialist manifests (``algorithm == "unet"``) route to the supervised
    segmentation adapter. Both paths are fully offline and hash-pinned, so the
    previous specialist remains a valid rollback target.
    """

    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("specialist run manifest root must be an object")
    if manifest.get("algorithm") == "unet":
        return _create_unet_specialist_adapter(manifest_file, manifest)
    return _create_patchcore_specialist_adapter(manifest_file, manifest)


def _create_patchcore_specialist_adapter(
    manifest_file: Path, manifest: Mapping[str, Any]
) -> PatchcoreServiceAdapterBridge:
    from .patchcore_specialist import PatchcoreConfig, _verify_backbone

    if manifest.get("status") != "completed" or manifest.get("schema_version") != "1.0":
        raise ValueError("specialist run manifest is not a completed v1 run")
    implementation = manifest.get("implementation")
    if not isinstance(implementation, Mapping) or implementation.get("algorithm") != "patchcore":
        raise ValueError("specialist run manifest is not PatchCore")
    artifacts = manifest.get("artifacts")
    calibration = manifest.get("calibration")
    raw_config = manifest.get("configuration")
    if not all(isinstance(value, Mapping) for value in (artifacts, calibration, raw_config)):
        raise ValueError("specialist manifest is missing runtime sections")
    checkpoint_value = artifacts.get("checkpoint")
    checkpoint_hash = artifacts.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not isinstance(checkpoint_hash, str):
        raise ValueError("specialist checkpoint identity is missing")
    checkpoint = Path(checkpoint_value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = manifest_file.parent / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file() or _file_sha256(checkpoint) != checkpoint_hash:
        raise ValueError("specialist checkpoint SHA-256 verification failed")
    if manifest.get("fit", {}).get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("specialist manifest checkpoint hashes disagree")

    config_payload = dict(raw_config)
    config_payload["layers"] = tuple(config_payload.get("layers", ()))
    config_payload["bbox_threshold_candidates"] = tuple(
        config_payload.get("bbox_threshold_candidates", ())
    )
    config = PatchcoreConfig(**config_payload)
    _verify_backbone(config)
    image_threshold = _finite_float(calibration.get("image_threshold"), "image_threshold")
    bbox_threshold = _finite_float(calibration.get("bbox_threshold"), "bbox_threshold")
    if calibration.get("test_labels_used") is not False:
        raise ValueError("specialist runtime threshold must not use test labels")

    # Force offline mode before importing timm/anomalib. The pinned backbone is
    # verified above, so any cache miss fails instead of starting a download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from anomalib.models import Patchcore

    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("pinned specialist requires MPS but MPS is unavailable")
    module = Patchcore(
        backbone=config.backbone,
        layers=config.layers,
        pre_trained=True,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        num_neighbors=config.num_neighbors,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("specialist checkpoint payload is invalid")
    if checkpoint_payload.get("source_manifest_sha256") != config.dataset_manifest_sha256:
        raise ValueError("specialist checkpoint dataset identity differs from manifest")
    if json.dumps(checkpoint_payload.get("config"), sort_keys=True) != json.dumps(
        dict(raw_config), sort_keys=True
    ):
        raise ValueError("specialist checkpoint config differs from run manifest")
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("specialist checkpoint state_dict is missing")
    model = module.model
    model.load_state_dict(state_dict)
    device = torch.device(config.device)
    model.to(device).eval()
    return PatchcoreServiceAdapterBridge(
        model=model,
        torch_module=torch,
        device=device,
        image_size=config.image_size,
        image_threshold=image_threshold,
        bbox_threshold=bbox_threshold,
        checkpoint_sha256=checkpoint_hash,
    )


def _create_unet_specialist_adapter(
    manifest_file: Path, manifest: Mapping[str, Any]
) -> UnetServiceAdapterBridge:
    """Load the phase 8 U-Net specialist from a signed-off phase 8 manifest."""

    from src.evaluation.localization_postprocess import PostprocessConfig
    from src.training.unet_segmentation import build_unet

    if manifest.get("status") != "completed" or manifest.get("schema_version") != "1.0":
        raise ValueError("phase 8 specialist manifest is not a completed v1 run")
    legacy_manifest = (
        manifest.get("phase") == "phase8"
        and manifest.get("kind") == "phase8_specialist_manifest"
    )
    revised_manifest = (
        manifest.get("phase") == "phase8.1"
        and manifest.get("kind") == "phase8_1_specialist_manifest"
    )
    if not legacy_manifest and not revised_manifest:
        raise ValueError("phase 8 specialist manifest kind is invalid")
    if legacy_manifest and manifest.get("test_labels_used_for_selection") is not False:
        raise ValueError("phase 8 specialist selection must not use test labels")
    if revised_manifest:
        if manifest.get("test_labels_used_for_training") is not False:
            raise ValueError("phase 8.1 specialist training must not use test labels")
        if manifest.get("test_labels_used_for_postprocess_selection") is not False:
            raise ValueError("phase 8.1 postprocess selection must not use test labels")
        if not isinstance(manifest.get("test_labels_used_for_route_comparison"), bool):
            raise ValueError("phase 8.1 route-comparison provenance is missing")
        if manifest.get("external_holdout") is not False:
            raise ValueError("phase 8.1 KSDD evidence cannot claim an external holdout")
        if manifest.get("production_ready") is not False:
            raise ValueError("phase 8.1 specialist is restricted to pilot serving")
    checkpoint_value = manifest.get("checkpoint")
    checkpoint_hash = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not isinstance(checkpoint_hash, str):
        raise ValueError("phase 8 specialist checkpoint identity is missing")
    checkpoint = Path(checkpoint_value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = manifest_file.parent / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file() or _file_sha256(checkpoint) != checkpoint_hash:
        raise ValueError("phase 8 specialist checkpoint SHA-256 verification failed")

    image_threshold = _finite_float(manifest.get("image_threshold"), "image_threshold")
    fusion_mode = manifest.get("fusion_mode")
    if fusion_mode not in {"max", "mean", "weighted"}:
        raise ValueError("phase 8 specialist fusion mode is invalid")
    postprocess_payload = manifest.get("postprocess_config")
    if not isinstance(postprocess_payload, Mapping):
        raise ValueError("phase 8 specialist postprocess config is missing")
    postprocess_config = PostprocessConfig(**dict(postprocess_payload))
    input_size = int(manifest.get("input_size", 0))
    tile_size = int(manifest.get("tile_size", 0))
    tile_stride = int(manifest.get("tile_stride", 0))
    if input_size <= 0 or tile_size <= 0 or tile_stride <= 0:
        raise ValueError("phase 8 specialist tile geometry is invalid")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    device_name = str(manifest.get("device", "mps"))
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("pinned specialist requires MPS but MPS is unavailable")
    device = torch.device(device_name)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("phase 8 specialist checkpoint payload is invalid")
    if checkpoint_payload.get("source_manifest_sha256") != manifest.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError("phase 8 specialist dataset identity differs from manifest")
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("phase 8 specialist checkpoint state_dict is missing")

    encoder = manifest.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("phase 8 specialist encoder identity is missing")

    class _EncoderAdapter:
        backbone_revision = str(encoder.get("revision"))
        backbone_bytes = int(encoder.get("bytes"))
        backbone_sha256 = str(encoder.get("sha256"))

    from src.inference.patchcore_tiled import _verify_backbone

    encoder_weights = _verify_backbone(_EncoderAdapter())
    model = build_unet(torch, encoder_weights)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return UnetServiceAdapterBridge(
        model=model,
        torch_module=torch,
        device=device,
        input_size=input_size,
        tile_size=tile_size,
        tile_stride=tile_stride,
        image_threshold=image_threshold,
        fusion_mode=str(fusion_mode),
        postprocess_config=postprocess_config,
        checkpoint_sha256=checkpoint_hash,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"specialist {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"specialist {name} must be finite and non-negative")
    return number
