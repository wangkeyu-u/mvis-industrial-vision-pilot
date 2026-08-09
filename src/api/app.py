"""FastAPI application factory and the mock-wired local entry point."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import time
import uuid
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.core.analyze_service import AnalyzeCommand, AnalyzeService
from src.core.config import ServiceSettings
from src.core.errors import ErrorCode, ServiceError
from src.core.image_validation import decode_image_data_url
from src.core.model_registry import ModelRegistry
from src.core.registry_factory import build_service_registry
from src.core.schemas import (
    SCHEMA_VERSION,
    AnalyzeOptions,
    AnalyzeResponse,
    AnalyzeTask,
    Base64AnalyzeRequest,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    ModelOpsActionRequest,
    ModelOpsActionResponse,
    VersionResponse,
)
from src.observability.context import bind_request_id, reset_request_id
from src.observability.logging import configure_logging, get_logger
from src.observability.resources import peak_memory_mb

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
MODELOPS_ACTOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


class RequestContextMiddleware:
    """Pure ASGI middleware that does not consume client disconnect events."""

    def __init__(
        self,
        app: ASGIApp,
        logger,  # type: ignore[no-untyped-def]
        registry: ModelRegistry,
    ) -> None:
        self.app = app
        self.logger = logger
        self.registry = registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers = dict(scope.get("headers", []))
        supplied = raw_headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else _new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        token = bind_request_id(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                if not any(name.lower() == b"x-request-id" for name, _ in headers):
                    headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            active = self.registry.active()
            provenance = active.provenance if active is not None else None
            self.logger.info(
                "http_request_completed",
                extra={
                    "model_id": active.model_id if active else None,
                    "adapter_id": active.adapter.identity.adapter if active else None,
                    "model_revision": (provenance.checkpoint_revision if provenance else None),
                    "config_fingerprint": provenance.config_fingerprint if provenance else None,
                    "config_artifact_sha256": (
                        provenance.config_artifact_sha256 if provenance else None
                    ),
                    "model_fingerprint": active.model_fingerprint if active else None,
                    "weight_hash": provenance.weight_hash if provenance else None,
                    "adapter_hash": provenance.adapter_hash if provenance else None,
                    "data_version": provenance.data_version if provenance else None,
                    "prompt_version": provenance.prompt_version if provenance else None,
                    "quality_status": active.quality_status.value if active else None,
                    "quality_accepted": active.quality_accepted if active else False,
                    "serving_tier": (
                        active.serving_tier.value if active and active.serving_tier else None
                    ),
                    "production_ready": self.registry.production_ready(),
                    "status": status_code,
                    "latency": {
                        "http_total_ms": max(0, round((time.perf_counter() - started) * 1000))
                    },
                    "memory_peak_mb": peak_memory_mb(),
                },
            )
            reset_request_id(token)


def create_app(
    settings: ServiceSettings | None = None,
    registry: ModelRegistry | None = None,
    *,
    emit_config_log: bool | None = None,
) -> FastAPI:
    service_settings = settings or ServiceSettings.load()
    model_registry = registry or build_service_registry(service_settings)
    configure_logging(service_settings.log_level)
    logger = get_logger()

    application = FastAPI(
        title="Lightweight Multimodal Vision API",
        version=service_settings.service_version,
        docs_url="/docs",
        redoc_url=None,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(service_settings.cors_allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )
    application.add_middleware(
        RequestContextMiddleware,
        logger=logger,
        registry=model_registry,
    )
    application.state.settings = service_settings
    application.state.registry = model_registry
    application.state.analyze_service = AnalyzeService(service_settings, model_registry)
    configured_model = model_registry.active()
    runtime_status = model_registry.runtime_status()

    def model_log_context(registration=None) -> dict[str, object]:  # type: ignore[no-untyped-def]
        selected = registration or model_registry.active()
        provenance = selected.provenance if selected is not None else None
        return {
            "model_id": selected.model_id if selected else None,
            "adapter_id": selected.adapter.identity.adapter if selected else None,
            "model_revision": provenance.checkpoint_revision if provenance else None,
            "config_fingerprint": provenance.config_fingerprint if provenance else None,
            "config_artifact_sha256": (provenance.config_artifact_sha256 if provenance else None),
            "model_fingerprint": selected.model_fingerprint if selected else None,
            "weight_hash": provenance.weight_hash if provenance else None,
            "adapter_hash": provenance.adapter_hash if provenance else None,
            "data_version": provenance.data_version if provenance else None,
            "prompt_version": provenance.prompt_version if provenance else None,
            "quality_status": selected.quality_status.value if selected else None,
            "quality_accepted": selected.quality_accepted if selected else False,
            "serving_tier": (
                selected.serving_tier.value if selected and selected.serving_tier else None
            ),
        }

    should_emit_config_log = (
        os.getenv("MVIS_SUPPRESS_CONFIG_LOG") != "1" if emit_config_log is None else emit_config_log
    )
    if should_emit_config_log:
        logger.info(
            "service_configured",
            extra={
                **model_log_context(configured_model),
                "quantization": (configured_model.quantization if configured_model else None),
                "runtime_mode": runtime_status["selected_mode"],
                "degraded": runtime_status["degraded"],
                "fallback_reason": runtime_status["fallback_reason"],
                "status": "ready" if model_registry.ready() else "not_ready",
                "production_ready": model_registry.production_ready(),
                "memory_peak_mb": peak_memory_mb(),
            },
        )

    @application.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", _new_request_id())
        error_runtime = model_registry.runtime_status()
        logger.warning(
            "request_failed",
            extra={
                **model_log_context(),
                "request_id": request_id,
                "status": exc.status_code,
                "error_code": exc.code.value,
                "runtime_mode": error_runtime["selected_mode"],
                "degraded": error_runtime["degraded"],
                "fallback_reason": error_runtime["fallback_reason"],
                "memory_peak_mb": peak_memory_mb(),
            },
        )
        body = ErrorResponse(
            request_id=request_id,
            error=ErrorBody(
                code=exc.code.value,
                message=exc.message,
                request_id=request_id,
                details=exc.details,
            ),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(mode="json"),
            headers={"X-Request-ID": request_id},
        )

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        fields = [str((error.get("loc") or ("request",))[-1]) for error in exc.errors()]
        if request.url.path.startswith("/v1/models/"):
            code = ErrorCode.MODELOPS_CONFLICT
        else:
            code = ErrorCode.INVALID_IMAGE if "image" in fields else ErrorCode.INVALID_QUERY
        return await service_error_handler(
            request,
            ServiceError(code, "request form fields are invalid", details={"fields": fields}),
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", _new_request_id())
        logger.exception(
            "unhandled_request_error",
            extra={
                **model_log_context(),
                "request_id": request_id,
                "status": 500,
                "error_code": ErrorCode.INTERNAL_ERROR.value,
                "memory_peak_mb": peak_memory_mb(),
            },
            exc_info=exc,
        )
        body = ErrorResponse(
            request_id=request_id,
            error=ErrorBody(
                code=ErrorCode.INTERNAL_ERROR.value,
                message="an unexpected internal error occurred",
                request_id=request_id,
            ),
        )
        return JSONResponse(
            status_code=500,
            content=body.model_dump(mode="json"),
            headers={"X-Request-ID": request_id},
        )

    @application.post(
        "/v1/analyze",
        response_model=AnalyzeResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            499: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/json": {"schema": Base64AnalyzeRequest.model_json_schema()}
                }
            }
        },
    )
    async def analyze(
        request: Request,
        image: Annotated[UploadFile | None, File()] = None,
        query: Annotated[str | None, Form()] = None,
        task: Annotated[str, Form()] = AnalyzeTask.INSPECT.value,
        model: Annotated[str, Form()] = "active",
        use_specialist: Annotated[bool, Form()] = False,
        options: Annotated[str | None, Form()] = None,
    ) -> AnalyzeResponse:
        media_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        image_content_type: str | None = None
        if media_type == "application/json":
            max_json_bytes = 4 * ((service_settings.max_image_bytes + 2) // 3) + 65536
            chunks: list[bytes] = []
            body_size = 0
            async for chunk in request.stream():
                body_size += len(chunk)
                if body_size > max_json_bytes:
                    raise ServiceError(ErrorCode.INVALID_IMAGE, "JSON request body is too large")
                chunks.append(chunk)
            try:
                json_request = Base64AnalyzeRequest.model_validate_json(b"".join(chunks))
            except ValidationError as exc:
                fields = [str((error.get("loc") or ("request",))[-1]) for error in exc.errors()]
                code = (
                    ErrorCode.INVALID_IMAGE
                    if any(field.startswith("image") for field in fields)
                    else ErrorCode.INVALID_QUERY
                )
                raise ServiceError(
                    code, "JSON request fields are invalid", details={"fields": fields}
                ) from exc
            image_payload, image_content_type = decode_image_data_url(
                json_request.image, service_settings
            )
            normalized_query = json_request.query
            parsed_task = json_request.task
            model = json_request.model
            use_specialist = json_request.use_specialist
            parsed_options = json_request.options
        else:
            if image is None:
                raise ServiceError(ErrorCode.INVALID_IMAGE, "image is required")
            image_payload = image
            normalized_query = (query or "").strip()
            if not 1 <= len(normalized_query) <= 1000:
                raise ServiceError(
                    ErrorCode.INVALID_QUERY, "query must contain 1 to 1000 characters"
                )
            try:
                parsed_task = AnalyzeTask(task)
            except ValueError as exc:
                raise ServiceError(
                    ErrorCode.INVALID_QUERY,
                    "task must be one of inspect, ground, extract, or vqa",
                ) from exc
            try:
                raw_options = json.loads(options) if options else {}
                if not isinstance(raw_options, dict):
                    raise TypeError("options must be an object")
                parsed_options = AnalyzeOptions.model_validate(raw_options)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                raise ServiceError(
                    ErrorCode.INVALID_QUERY,
                    "options must be a JSON object containing only supported fields",
                ) from exc

        response, log_fields = await application.state.analyze_service.analyze(
            AnalyzeCommand(
                image=image_payload,
                image_content_type=image_content_type,
                query=normalized_query,
                task=parsed_task,
                model=model,
                use_specialist=use_specialist,
                options=parsed_options,
            ),
            request.state.request_id,
            cancellation_check=request.is_disconnected,
        )
        logger.info(
            "analyze_completed",
            extra=log_fields
            | {
                "request_id": request.state.request_id,
                "status": 200,
                "error_code": None,
                "memory_peak_mb": peak_memory_mb(),
            },
        )
        return response

    @application.get("/health/live", response_model=HealthResponse)
    async def live(request: Request) -> HealthResponse:
        return HealthResponse(status="live", request_id=request.state.request_id)

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
    )
    async def ready(request: Request):  # type: ignore[no-untyped-def]
        is_ready = model_registry.ready()
        body = HealthResponse(
            status="ready" if is_ready else "not_ready",
            request_id=request.state.request_id,
            details=model_registry.statuses(),
        )
        if is_ready:
            return body
        return JSONResponse(status_code=503, content=body.model_dump(mode="json"))

    @application.get("/version", response_model=VersionResponse)
    async def version() -> VersionResponse:
        active = model_registry.active()
        return VersionResponse(
            service=service_settings.service_name,
            code=service_settings.code_version,
            api="v1",
            schema_version=SCHEMA_VERSION,
            model=active.identity if active else None,
            provenance=active.provenance if active else None,
            quality_status=active.quality_status if active else None,
            quality_accepted=active.quality_accepted if active else False,
            serving_tier=active.serving_tier if active else None,
            runtime=model_registry.runtime_status()
            | {
                "runtime_ready": model_registry.ready(),
                "production_ready": model_registry.production_ready(),
            },
        )

    @application.get("/v1/models")
    async def models() -> dict[str, object]:
        return model_registry.statuses()

    def authorize_modelops(token: str | None, actor: str | None) -> str:
        configured_token = service_settings.modelops_token
        if configured_token is None:
            raise ServiceError(
                ErrorCode.MODELOPS_DISABLED,
                "ModelOps mutations are disabled until MVIS_MODELOPS_TOKEN is set",
            )
        if token is None or not hmac.compare_digest(token, configured_token):
            raise ServiceError(
                ErrorCode.MODELOPS_UNAUTHORIZED,
                "valid ModelOps credentials are required",
            )
        normalized_actor = (actor or "").strip()
        if not MODELOPS_ACTOR_PATTERN.fullmatch(normalized_actor):
            raise ServiceError(
                ErrorCode.MODELOPS_CONFLICT,
                "X-ModelOps-Actor must contain a safe operator identifier",
            )
        return normalized_actor

    def run_modelops_action(
        *,
        action: str,
        request: Request,
        payload: ModelOpsActionRequest,
        token: str | None,
        actor_header: str | None,
        model_id: str | None = None,
    ) -> ModelOpsActionResponse:
        actor = authorize_modelops(token, actor_header)
        if model_id is not None and not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ServiceError(
                ErrorCode.MODELOPS_CONFLICT,
                "model_id contains unsupported characters",
            )
        try:
            if action == "validate" and model_id is not None:
                audit = model_registry.validate_candidate(
                    model_id,
                    actor=actor,
                    request_id=request.state.request_id,
                    reason=payload.reason,
                )
            elif action == "activate" and model_id is not None:
                audit = model_registry.activate(
                    model_id,
                    actor=actor,
                    request_id=request.state.request_id,
                    reason=payload.reason,
                    expected_active_model_id=payload.expected_active_model_id,
                    expected_active_fingerprint=payload.expected_active_fingerprint,
                    serving_tier=payload.serving_tier,
                )
            elif action == "rollback":
                audit = model_registry.rollback(
                    actor=actor,
                    request_id=request.state.request_id,
                    reason=payload.reason,
                    expected_active_model_id=payload.expected_active_model_id,
                    expected_active_fingerprint=payload.expected_active_fingerprint,
                    serving_tier=payload.serving_tier,
                )
            elif action == "retire" and model_id is not None:
                audit = model_registry.retire(
                    model_id,
                    actor=actor,
                    request_id=request.state.request_id,
                    reason=payload.reason,
                )
            else:  # pragma: no cover - routes pass only fixed actions
                raise ValueError("unsupported ModelOps action")
        except ValueError as exc:
            raise ServiceError(
                ErrorCode.MODELOPS_CONFLICT,
                str(exc),
                details={"action": action, "model_id": model_id},
            ) from exc

        audit_status = audit.public_status()
        logger.info(
            "modelops_lifecycle_changed",
            extra={
                **model_log_context(model_registry.active()),
                "request_id": request.state.request_id,
                "model_id": audit.model_id,
                "status": 200,
                "audit": audit_status,
                "memory_peak_mb": peak_memory_mb(),
            },
        )
        return ModelOpsActionResponse(
            request_id=request.state.request_id,
            action=action,
            audit=audit_status,
            registry=model_registry.statuses(),
        )

    @application.post(
        "/v1/models/{model_id}/validate",
        response_model=ModelOpsActionResponse,
    )
    async def validate_model(
        model_id: str,
        payload: ModelOpsActionRequest,
        request: Request,
        token: Annotated[str | None, Header(alias="X-ModelOps-Token")] = None,
        actor: Annotated[str | None, Header(alias="X-ModelOps-Actor")] = None,
    ) -> ModelOpsActionResponse:
        return run_modelops_action(
            action="validate",
            request=request,
            payload=payload,
            token=token,
            actor_header=actor,
            model_id=model_id,
        )

    @application.post(
        "/v1/models/{model_id}/activate",
        response_model=ModelOpsActionResponse,
    )
    async def activate_model(
        model_id: str,
        payload: ModelOpsActionRequest,
        request: Request,
        token: Annotated[str | None, Header(alias="X-ModelOps-Token")] = None,
        actor: Annotated[str | None, Header(alias="X-ModelOps-Actor")] = None,
    ) -> ModelOpsActionResponse:
        return run_modelops_action(
            action="activate",
            request=request,
            payload=payload,
            token=token,
            actor_header=actor,
            model_id=model_id,
        )

    @application.post(
        "/v1/models/rollback",
        response_model=ModelOpsActionResponse,
    )
    async def rollback_model(
        payload: ModelOpsActionRequest,
        request: Request,
        token: Annotated[str | None, Header(alias="X-ModelOps-Token")] = None,
        actor: Annotated[str | None, Header(alias="X-ModelOps-Actor")] = None,
    ) -> ModelOpsActionResponse:
        return run_modelops_action(
            action="rollback",
            request=request,
            payload=payload,
            token=token,
            actor_header=actor,
        )

    @application.post(
        "/v1/models/{model_id}/retire",
        response_model=ModelOpsActionResponse,
    )
    async def retire_model(
        model_id: str,
        payload: ModelOpsActionRequest,
        request: Request,
        token: Annotated[str | None, Header(alias="X-ModelOps-Token")] = None,
        actor: Annotated[str | None, Header(alias="X-ModelOps-Actor")] = None,
    ) -> ModelOpsActionResponse:
        return run_modelops_action(
            action="retire",
            request=request,
            payload=payload,
            token=token,
            actor_header=actor,
            model_id=model_id,
        )

    @application.get("/v1/models/audit")
    async def model_audit(
        request: Request,
        limit: int = 100,
        token: Annotated[str | None, Header(alias="X-ModelOps-Token")] = None,
        actor: Annotated[str | None, Header(alias="X-ModelOps-Actor")] = None,
    ) -> dict[str, object]:
        authorize_modelops(token, actor)
        return {
            "request_id": request.state.request_id,
            "audit": model_registry.audit_records(limit=limit),
        }

    return application


class LazyServiceApplication:
    """Delay default app construction until the first ASGI event.

    The CLI supplies an already-preflighted app directly to Uvicorn. Keeping the
    import-level fallback lazy prevents a second real-model load merely from
    importing ``create_app``.
    """

    def __init__(self) -> None:
        self._application: FastAPI | None = None
        self._lock: asyncio.Lock | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._application is None:
            if self._lock is None:
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._application is None:
                    self._application = create_app()
        await self._application(scope, receive, send)


app = LazyServiceApplication()
