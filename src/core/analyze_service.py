"""Application service coordinating validation, resources, and adapter calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import ValidationError

from src.core.config import ServiceSettings
from src.core.errors import ErrorCode, ServiceError
from src.core.image_validation import validate_image_bytes, validate_upload
from src.core.model_registry import AdapterRequest, ModelRegistry
from src.core.schemas import (
    AnalyzeOptions,
    AnalyzeResponse,
    AnalyzeTask,
    LatencyBreakdown,
    ModelOutput,
)


@dataclass(frozen=True, slots=True)
class AnalyzeCommand:
    image: object
    image_content_type: str | None
    query: str
    task: AnalyzeTask
    model: str
    use_specialist: bool
    options: AnalyzeOptions


class AnalyzeService:
    def __init__(self, settings: ServiceSettings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._capacity = asyncio.Semaphore(settings.concurrency_limit)

    async def analyze(
        self,
        command: AnalyzeCommand,
        request_id: str,
        cancellation_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[AnalyzeResponse, dict[str, object]]:
        total_started = time.perf_counter()
        preprocess_started = total_started
        try:
            if isinstance(command.image, bytes):
                image_bytes, image_meta = validate_image_bytes(
                    command.image, command.image_content_type or "", self.settings
                )
            else:
                image_bytes, image_meta = await validate_upload(
                    command.image, self.settings
                )  # type: ignore[arg-type]
        except MemoryError as exc:
            raise ServiceError(
                ErrorCode.RESOURCE_EXHAUSTED,
                "image preprocessing resources are exhausted",
            ) from exc
        preprocess_ms = _elapsed_ms(preprocess_started)
        registration = self.registry.resolve(command.model)

        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._capacity.acquire(),
                    timeout=self.settings.concurrency_wait_seconds,
                )
                acquired = True
            except TimeoutError as exc:
                raise ServiceError(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "inference capacity is currently exhausted; retry later",
                ) from exc

            inference_started = time.perf_counter()
            try:
                raw_output = await _run_with_cancellation(
                    registration.adapter.analyze(
                        AdapterRequest(
                            image_bytes=image_bytes,
                            image=image_meta,
                            request_id=request_id,
                            query=command.query,
                            task=command.task,
                            use_specialist=command.use_specialist,
                            options=command.options,
                        )
                    ),
                    timeout_seconds=self.settings.inference_timeout_seconds,
                    cancellation_check=cancellation_check,
                    poll_seconds=self.settings.disconnect_poll_seconds,
                )
            except TimeoutError as exc:
                raise ServiceError(
                    ErrorCode.INFERENCE_TIMEOUT,
                    f"inference exceeded {self.settings.inference_timeout_seconds:g}s",
                ) from exc
            except MemoryError as exc:
                raise ServiceError(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "inference resources are exhausted",
                ) from exc
            inference_ms = _elapsed_ms(inference_started)

            validation_started = time.perf_counter()
            try:
                output = ModelOutput.model_validate(raw_output)
                _validate_bounds(output, image_meta.width, image_meta.height)
            except (ValidationError, ValueError) as exc:
                raise ServiceError(
                    ErrorCode.OUTPUT_VALIDATION_FAILED,
                    "model output failed schema or coordinate validation",
                ) from exc
            validation_ms = _elapsed_ms(validation_started)
        finally:
            if acquired:
                self._capacity.release()

        total_ms = _elapsed_ms(total_started)
        timing = LatencyBreakdown(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            validation_ms=validation_ms,
        )
        response = AnalyzeResponse(
            request_id=request_id,
            model=registration.adapter.identity,
            result=output.result,
            objects=output.objects,
            reason=output.reason,
            uncertain=output.uncertain,
            latency_ms=total_ms,
            latency=timing,
            timing=timing,
            warnings=output.warnings,
        )
        log_fields: dict[str, object] = {
            "model_id": registration.model_id,
            "adapter_id": registration.adapter.identity.adapter,
            "quantization": registration.quantization,
            "task": command.task.value,
            "image_shape": image_meta.log_shape,
            "latency": response.latency.model_dump() | {"total_ms": total_ms},
        }
        runtime_status = self.registry.runtime_status()
        log_fields.update(
            runtime_mode=runtime_status["selected_mode"],
            degraded=runtime_status["degraded"],
            fallback_reason=runtime_status["fallback_reason"],
        )
        return response, log_fields


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


async def _run_with_cancellation(
    operation: Awaitable[object],
    *,
    timeout_seconds: float,
    cancellation_check: Callable[[], Awaitable[bool]] | None,
    poll_seconds: float,
) -> object:
    task = asyncio.create_task(operation)
    try:
        if cancellation_check is None:
            return await asyncio.wait_for(task, timeout=timeout_seconds)

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError
            done, _ = await asyncio.wait({task}, timeout=min(poll_seconds, remaining))
            if done:
                return await task
            if await cancellation_check():
                raise ServiceError(
                    ErrorCode.REQUEST_CANCELLED,
                    "client disconnected before inference completed",
                )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _validate_bounds(output: ModelOutput, width: int, height: int) -> None:
    for evidence in output.objects:
        x1, y1, x2, y2 = evidence.bbox
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError("bbox lies outside the original image")
