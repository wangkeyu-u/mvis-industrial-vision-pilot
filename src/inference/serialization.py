"""Machine-readable serialization for the framework-neutral model result."""

from __future__ import annotations

from typing import Any

from .contracts import ModelResult


def model_result_to_dict(
    result: ModelResult, *, include_raw_text: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "result": result.decision.value,
        "objects": [
            {
                "label": item.label,
                "bbox": list(item.bbox),
                "confidence": item.confidence,
                "source": item.source.value,
            }
            for item in result.objects
        ],
        "reason": result.reason,
        "uncertain": result.uncertain,
        "refusal": (
            {
                "code": result.refusal.code.value,
                "message": result.refusal.message,
                "review_required": result.refusal.review_required,
            }
            if result.refusal is not None
            else None
        ),
        "provenance": {
            "model_id": result.provenance.model_id,
            "model_revision": result.provenance.model_revision,
            "backend": result.provenance.backend,
            "config_fingerprint": result.provenance.config_fingerprint,
            "adapter_id": result.provenance.adapter_id,
            "seed": result.provenance.seed,
            "deterministic": result.provenance.deterministic,
            "extra": dict(result.provenance.extra),
        },
        "warnings": list(result.warnings),
    }
    if include_raw_text:
        payload["raw_text"] = result.raw_text
    return payload
