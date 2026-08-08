"""Thin compatibility bridge to the existing async service-side protocol.

The framework-neutral adapter remains the source of model behavior.  This
module only translates existing ``src.core`` request/result objects and keeps
the blocking MLX call off the event loop.
"""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

from src.core.errors import ErrorCode, ServiceError
from src.core.model_registry import AdapterRequest
from src.core.schemas import EvidenceObject, ModelIdentity, ModelOutput, ObjectSource

from .base import GenerationBackend
from .config import load_model_config
from .contracts import GenerationConfig, ModelRequest, Task
from .errors import ModelNotReadyError, ModelOutputError
from .mlx_backend import MlxVlmBackend
from .qwen3_vl import Qwen3VLAdapter


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
