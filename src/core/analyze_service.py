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
from src.core.model_registry import (
    AdapterRequest,
    ModelRegistration,
    ModelRegistry,
)
from src.core.schemas import (
    AnalysisMode,
    AnalyzeOptions,
    AnalyzeResponse,
    AnalyzeTask,
    EvidenceObject,
    ImageMetadata,
    LatencyBreakdown,
    ModelOutput,
    SpecialistEvidence,
    SpecialistOutput,
)
from src.core.specialist_runtime import (
    HeatmapArtifactStore,
    SpecialistRegistration,
    SpecialistRequest,
    conservative_fuse,
    specialist_only_output,
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
    analysis_mode: AnalysisMode = AnalysisMode.VLM_ONLY


class AnalyzeService:
    def __init__(
        self,
        settings: ServiceSettings,
        registry: ModelRegistry,
        heatmap_store: HeatmapArtifactStore | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.heatmap_store = heatmap_store or HeatmapArtifactStore(
            max_items=settings.heatmap_max_items,
            max_bytes=settings.heatmap_max_bytes,
            ttl_seconds=settings.heatmap_ttl_seconds,
            max_pixels=settings.heatmap_max_pixels,
        )
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
                image_bytes, image_meta = await validate_upload(command.image, self.settings)  # type: ignore[arg-type]
        except MemoryError as exc:
            raise ServiceError(
                ErrorCode.RESOURCE_EXHAUSTED,
                "image preprocessing resources are exhausted",
            ) from exc
        preprocess_ms = _elapsed_ms(preprocess_started)
        registration = (
            self.registry.resolve(command.model)
            if command.analysis_mode in {AnalysisMode.VLM_ONLY, AnalysisMode.FUSED}
            else None
        )
        specialist_registration = (
            self.registry.resolve_specialist()
            if command.analysis_mode in {AnalysisMode.SPECIALIST_ONLY, AnalysisMode.FUSED}
            else None
        )

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
                raw_model_output, raw_specialist_output = await _run_with_cancellation(
                    _invoke_adapters(
                        command=command,
                        request_id=request_id,
                        image_bytes=image_bytes,
                        image_meta=image_meta,
                        registration=registration,
                        specialist_registration=specialist_registration,
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
                model_output = (
                    ModelOutput.model_validate(raw_model_output)
                    if raw_model_output is not None
                    else None
                )
                specialist_output = (
                    SpecialistOutput.model_validate(raw_specialist_output)
                    if raw_specialist_output is not None
                    else None
                )
                if command.analysis_mode is AnalysisMode.VLM_ONLY:
                    assert model_output is not None
                    output = model_output
                    human_review_required = output.uncertain
                elif command.analysis_mode is AnalysisMode.SPECIALIST_ONLY:
                    assert specialist_output is not None
                    output = specialist_only_output(specialist_output)
                    human_review_required = False
                else:
                    assert model_output is not None and specialist_output is not None
                    output, human_review_required = conservative_fuse(
                        model_output,
                        specialist_output,
                        iou_threshold=self.settings.fusion_iou_threshold,
                    )
                _validate_bounds(output, image_meta.width, image_meta.height)
                if specialist_output is not None:
                    _validate_objects_bounds(
                        specialist_output.objects,
                        image_meta.width,
                        image_meta.height,
                    )
                    heatmap_artifact = self.heatmap_store.add(
                        specialist_output.heatmap_png
                    )
                else:
                    heatmap_artifact = None
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
        selected_identity = (
            registration.identity
            if registration is not None
            else specialist_registration.adapter.identity
        )
        selected_provenance = (
            registration.provenance
            if registration is not None
            else specialist_registration.provenance
        )
        assert selected_provenance is not None
        quality_status = self.registry.mode_quality_status(command.analysis_mode)
        quality_accepted = self.registry.mode_quality_accepted(command.analysis_mode)
        serving_tier = self.registry.mode_serving_tier(command.analysis_mode)
        specialist_evidence = None
        if specialist_output is not None and specialist_registration is not None:
            specialist_evidence = SpecialistEvidence(
                specialist_id=specialist_registration.specialist_id,
                revision=specialist_registration.adapter.identity.revision,
                score=specialist_output.score,
                threshold=specialist_output.threshold,
                detected=specialist_output.detected,
                source=specialist_output.source,
                objects=specialist_output.objects,
                heatmap=heatmap_artifact,
                provenance=specialist_registration.provenance,
                quality_status=specialist_registration.quality_status,
                quality_accepted=specialist_registration.quality_accepted,
            )
        response = AnalyzeResponse(
            request_id=request_id,
            analysis_mode=command.analysis_mode,
            model=selected_identity,
            provenance=selected_provenance,
            quality_status=quality_status,
            quality_accepted=quality_accepted,
            serving_tier=serving_tier,
            specialist=specialist_evidence,
            human_review_required=human_review_required,
            result=output.result,
            objects=output.objects,
            reason=output.reason,
            uncertain=output.uncertain,
            latency_ms=total_ms,
            latency=timing,
            timing=timing,
            warnings=list(
                dict.fromkeys(
                    [
                        *output.warnings,
                        f"quality:{quality_status.value}",
                        f"serving_tier:{serving_tier.value}",
                    ]
                )
            ),
        )
        log_fields: dict[str, object] = {
            "model_id": (
                registration.model_id
                if registration is not None
                else specialist_registration.specialist_id
            ),
            "adapter_id": selected_identity.adapter,
            "model_revision": selected_provenance.checkpoint_revision,
            "config_fingerprint": selected_provenance.config_fingerprint,
            "config_artifact_sha256": selected_provenance.config_artifact_sha256,
            "model_fingerprint": (
                registration.model_fingerprint if registration is not None else None
            ),
            "weight_hash": selected_provenance.weight_hash,
            "adapter_hash": selected_provenance.adapter_hash,
            "data_version": selected_provenance.data_version,
            "prompt_version": selected_provenance.prompt_version,
            "quality_status": quality_status.value,
            "quality_accepted": quality_accepted,
            "serving_tier": serving_tier.value,
            "production_ready": self.registry.production_ready(command.analysis_mode),
            "quantization": registration.quantization if registration is not None else None,
            "analysis_mode": command.analysis_mode.value,
            "specialist_id": (
                specialist_registration.specialist_id
                if specialist_registration is not None
                else None
            ),
            "specialist_source": (
                specialist_output.source.value if specialist_output is not None else None
            ),
            "specialist_revision": (
                specialist_registration.provenance.checkpoint_revision
                if specialist_registration is not None
                else None
            ),
            "specialist_config_fingerprint": (
                specialist_registration.provenance.config_fingerprint
                if specialist_registration is not None
                else None
            ),
            "specialist_weight_hash": (
                specialist_registration.provenance.weight_hash
                if specialist_registration is not None
                else None
            ),
            "specialist_data_version": (
                specialist_registration.provenance.data_version
                if specialist_registration is not None
                else None
            ),
            "specialist_quality_status": (
                specialist_registration.quality_status.value
                if specialist_registration is not None
                else None
            ),
            "specialist_quality_accepted": (
                specialist_registration.quality_accepted
                if specialist_registration is not None
                else False
            ),
            "specialist_score": (
                specialist_output.score if specialist_output is not None else None
            ),
            "specialist_threshold": (
                specialist_output.threshold if specialist_output is not None else None
            ),
            "human_review_required": human_review_required,
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


async def _invoke_adapters(
    *,
    command: AnalyzeCommand,
    request_id: str,
    image_bytes: bytes,
    image_meta: ImageMetadata,
    registration: ModelRegistration | None,
    specialist_registration: SpecialistRegistration | None,
) -> tuple[object | None, object | None]:
    model_request = AdapterRequest(
        image_bytes=image_bytes,
        image=image_meta,
        request_id=request_id,
        query=command.query,
        task=command.task,
        use_specialist=(
            command.use_specialist
            if command.analysis_mode is AnalysisMode.VLM_ONLY
            else False
        ),
        options=command.options,
    )
    specialist_request = SpecialistRequest(
        image_bytes=image_bytes,
        image=image_meta,
        request_id=request_id,
        query=command.query,
        task=command.task,
        options=command.options,
    )
    if command.analysis_mode is AnalysisMode.VLM_ONLY:
        assert registration is not None
        return await registration.adapter.analyze(model_request), None
    if command.analysis_mode is AnalysisMode.SPECIALIST_ONLY:
        assert specialist_registration is not None
        return None, await specialist_registration.adapter.detect(specialist_request)
    assert registration is not None and specialist_registration is not None
    model_result, specialist_result = await asyncio.gather(
        registration.adapter.analyze(model_request),
        specialist_registration.adapter.detect(specialist_request),
    )
    return model_result, specialist_result


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
    _validate_objects_bounds(output.objects, width, height)


def _validate_objects_bounds(
    objects: list[EvidenceObject], width: int, height: int
) -> None:
    for evidence in objects:
        x1, y1, x2, y2 = evidence.bbox
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError("bbox lies outside the original image")
