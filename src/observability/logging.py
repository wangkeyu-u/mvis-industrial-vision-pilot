"""JSON logging with privacy-safe, schema-stable fields."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from src.observability.context import request_id_context

REQUIRED_FIELDS = (
    "model_id",
    "adapter_id",
    "quantization",
    "runtime_mode",
    "degraded",
    "fallback_reason",
    "task",
    "image_shape",
    "latency",
    "memory_peak",
    "memory_peak_mb",
    "status",
    "error_code",
    "audit",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_context.get()),
        }
        for field in REQUIRED_FIELDS:
            fallback = getattr(record, "memory_peak_mb", None) if field == "memory_peak" else None
            payload[field] = getattr(record, field, fallback)
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    logger = logging.getLogger("mvis")
    logger.setLevel(level.upper())
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(JsonFormatter())


def get_logger() -> logging.Logger:
    return logging.getLogger("mvis")
