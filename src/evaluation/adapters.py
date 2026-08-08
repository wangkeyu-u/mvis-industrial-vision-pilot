"""Normalize algorithm and HTTP outputs into one evaluation contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .metrics import validate_prediction_payload

INVALID_RESULT = "__invalid__"


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


@dataclass(frozen=True)
class NormalizedObject:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float | None = None
    source: str | None = None

    def in_bounds(self, width: int, height: int) -> bool:
        x1, y1, x2, y2 = self.bbox
        return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"label": self.label, "bbox": list(self.bbox)}
        if self.confidence is not None:
            value["confidence"] = self.confidence
        if self.source is not None:
            value["source"] = self.source
        return value


@dataclass(frozen=True)
class NormalizedPrediction:
    result: str
    objects: tuple[NormalizedObject, ...]
    reason: str
    uncertain: bool
    refusal_code: str | None
    schema_valid: bool
    source_kind: str
    request_id: str | None = None
    model_id: str | None = None
    parse_error: str | None = None

    @classmethod
    def invalid(cls, source_kind: str, message: str) -> "NormalizedPrediction":
        return cls(
            result=INVALID_RESULT,
            objects=(),
            reason="invalid structured output",
            uncertain=False,
            refusal_code=None,
            schema_valid=False,
            source_kind=source_kind,
            parse_error=message,
        )

    def core_payload(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "objects": [item.to_dict() for item in self.objects],
            "reason": self.reason,
            "uncertain": self.uncertain,
        }

    def boundary_errors(self, width: int, height: int) -> tuple[int, ...]:
        return tuple(
            index for index, item in enumerate(self.objects) if not item.in_bounds(width, height)
        )

    def to_dict(self) -> dict[str, Any]:
        return self.core_payload() | {
            "refusal_code": self.refusal_code,
            "schema_valid": self.schema_valid,
            "source_kind": self.source_kind,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "parse_error": self.parse_error,
        }


def _parse_mapping(payload: str | Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg}"
    else:
        value = payload
    if not isinstance(value, Mapping):
        return None, "structured output must be a JSON object"
    return value, None


def _objects_from_mapping(value: Mapping[str, Any]) -> tuple[NormalizedObject, ...]:
    objects = []
    raw_objects = value.get("objects", [])
    if not isinstance(raw_objects, list):
        return ()
    for item in raw_objects:
        if not isinstance(item, Mapping):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            coordinates = tuple(float(coordinate) for coordinate in bbox)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            continue
        confidence = item.get("confidence")
        objects.append(
            NormalizedObject(
                label=str(item.get("label", "")),
                bbox=coordinates,
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                source=str(_enum_value(item["source"])) if item.get("source") is not None else None,
            )
        )
    return tuple(objects)


def _refusal_from_mapping(value: Mapping[str, Any]) -> str | None:
    refusal = value.get("refusal")
    if isinstance(refusal, Mapping) and refusal.get("code") is not None:
        return str(_enum_value(refusal["code"]))
    refusal_code = value.get("refusal_code")
    if refusal_code is not None:
        return str(_enum_value(refusal_code))
    warnings = value.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            if isinstance(warning, str) and warning.startswith("refusal:"):
                return warning.partition(":")[2] or None
    return None


def normalize_model_result(result: object) -> NormalizedPrediction:
    """Consume a live ``src.inference.contracts.ModelResult`` by stable attributes."""

    try:
        decision = str(_enum_value(getattr(result, "decision")))
        raw_objects = getattr(result, "objects")
        objects = tuple(
            NormalizedObject(
                label=item.label,
                bbox=tuple(float(coordinate) for coordinate in item.bbox),
                confidence=float(item.confidence),
                source=str(_enum_value(item.source)),
            )
            for item in raw_objects
        )
        refusal = getattr(result, "refusal")
        refusal_code = str(_enum_value(refusal.code)) if refusal is not None else None
        provenance = getattr(result, "provenance")
        prediction = NormalizedPrediction(
            result=decision,
            objects=objects,
            reason=str(getattr(result, "reason")),
            uncertain=bool(getattr(result, "uncertain")),
            refusal_code=refusal_code,
            schema_valid=True,
            source_kind="model_result",
            model_id=str(getattr(provenance, "model_id")),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return NormalizedPrediction.invalid("model_result", f"invalid ModelResult: {exc}")
    if not validate_prediction_payload(prediction.core_payload()):
        return NormalizedPrediction.invalid("model_result", "ModelResult violates evaluation schema")
    return prediction


def normalize_model_result_payload(
    payload: str | Mapping[str, Any],
) -> NormalizedPrediction:
    """Consume the JSON form produced by ``model_result_to_dict``."""

    value, error = _parse_mapping(payload)
    if value is None:
        return NormalizedPrediction.invalid("model_result", error or "invalid payload")
    schema_valid = (
        validate_prediction_payload(value)
        and isinstance(value.get("provenance"), Mapping)
        and isinstance(value["provenance"].get("model_id"), str)
        and bool(value["provenance"]["model_id"].strip())
    )
    provenance = value.get("provenance")
    model_id = provenance.get("model_id") if isinstance(provenance, Mapping) else None
    return NormalizedPrediction(
        result=str(value.get("result", INVALID_RESULT)),
        objects=_objects_from_mapping(value),
        reason=str(value.get("reason", "invalid structured output")),
        uncertain=value.get("uncertain") if isinstance(value.get("uncertain"), bool) else False,
        refusal_code=_refusal_from_mapping(value),
        schema_valid=schema_valid,
        source_kind="model_result",
        model_id=str(model_id) if model_id is not None else None,
        parse_error=None if schema_valid else "serialized ModelResult violates evaluation schema",
    )


def normalize_analyze_response(
    payload: str | Mapping[str, Any],
) -> NormalizedPrediction:
    """Consume a successful backend ``POST /v1/analyze`` JSON response."""

    value, error = _parse_mapping(payload)
    if value is None:
        return NormalizedPrediction.invalid("api", error or "invalid payload")
    envelope_fields = {
        "schema_version",
        "request_id",
        "model",
        "result",
        "objects",
        "reason",
        "uncertain",
        "latency_ms",
        "warnings",
    }
    model = value.get("model")
    envelope_valid = (
        envelope_fields.issubset(value)
        and isinstance(value.get("schema_version"), str)
        and bool(value["schema_version"].strip())
        and isinstance(value.get("request_id"), str)
        and bool(value["request_id"].strip())
        and isinstance(model, Mapping)
        and isinstance(model.get("base"), str)
        and bool(model["base"].strip())
        and isinstance(value.get("latency_ms"), int)
        and not isinstance(value["latency_ms"], bool)
        and value["latency_ms"] >= 0
        and isinstance(value.get("warnings"), list)
        and all(isinstance(warning, str) for warning in value["warnings"])
    )
    schema_valid = validate_prediction_payload(value) and envelope_valid
    model_id = model.get("base") if isinstance(model, Mapping) else None
    return NormalizedPrediction(
        result=str(value.get("result", INVALID_RESULT)),
        objects=_objects_from_mapping(value),
        reason=str(value.get("reason", "invalid structured output")),
        uncertain=value.get("uncertain") if isinstance(value.get("uncertain"), bool) else False,
        refusal_code=_refusal_from_mapping(value),
        schema_valid=schema_valid,
        source_kind="api",
        request_id=str(value["request_id"]) if isinstance(value.get("request_id"), str) else None,
        model_id=str(model_id) if model_id is not None else None,
        parse_error=None if schema_valid else "analyze response violates evaluation schema",
    )
