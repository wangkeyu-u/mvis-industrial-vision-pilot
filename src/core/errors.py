"""Stable service errors shared by the API and inference boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_IMAGE = "INVALID_IMAGE"
    INVALID_QUERY = "INVALID_QUERY"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    REQUEST_CANCELLED = "REQUEST_CANCELLED"
    MODELOPS_UNAUTHORIZED = "MODELOPS_UNAUTHORIZED"
    MODELOPS_DISABLED = "MODELOPS_DISABLED"
    MODELOPS_CONFLICT = "MODELOPS_CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_IMAGE: 400,
    ErrorCode.INVALID_QUERY: 400,
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.MODEL_NOT_READY: 503,
    ErrorCode.INFERENCE_TIMEOUT: 504,
    ErrorCode.OUTPUT_VALIDATION_FAILED: 422,
    ErrorCode.RESOURCE_EXHAUSTED: 503,
    ErrorCode.REQUEST_CANCELLED: 499,
    ErrorCode.MODELOPS_UNAUTHORIZED: 401,
    ErrorCode.MODELOPS_DISABLED: 503,
    ErrorCode.MODELOPS_CONFLICT: 409,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ServiceError(Exception):
    """An expected failure safe to expose to callers."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = ERROR_HTTP_STATUS[code]
