"""Qwen3-VL adapter implementing the stable compliance-review contract."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .base import FusionPolicy, GenerationBackend, ModelAdapter, SpecialistAdapter
from .config import ModelConfig
from .contracts import (
    BackendRequest,
    Decision,
    DetectedObject,
    EvidenceSource,
    GenerationConfig,
    ModelProvenance,
    ModelRequest,
    ModelResult,
    RefusalCode,
    RefusalSignal,
)
from .errors import ModelOutputError
from .prompting import build_compliance_prompt


class Qwen3VLAdapter(ModelAdapter):
    def __init__(
        self,
        config: ModelConfig,
        backend: GenerationBackend,
        specialist: SpecialistAdapter | None = None,
        fusion_policy: FusionPolicy | None = None,
    ) -> None:
        if config.family != "qwen3_vl":
            raise ValueError(f"Qwen3VLAdapter cannot load family {config.family!r}")
        self.config = config
        self.backend = backend
        self.specialist = specialist
        self.fusion_policy = fusion_policy

    @property
    def ready(self) -> bool:
        return self.backend.ready

    def load(self) -> None:
        self.backend.load()

    def analyze(self, request: ModelRequest) -> ModelResult:
        self._validate_limits(request)
        generation = self._effective_generation(request.generation or self.config.generation)
        prompt = build_compliance_prompt(request)
        response = self.backend.generate(
            BackendRequest(
                image=request.image,
                prompt=prompt,
                generation=generation,
                request_id=request.request_id,
            )
        )
        payload = self._parse_json(response.text)
        result = self._to_result(
            payload, request, response.text, response.metadata, generation
        )
        if not request.use_specialist:
            return result
        if self.specialist is None or not self.specialist.ready:
            return replace(
                result,
                warnings=result.warnings + ("specialist_requested_but_unavailable",),
            )
        if self.fusion_policy is None:
            return replace(
                result,
                warnings=result.warnings + ("specialist_fusion_policy_not_configured",),
            )
        return self.fusion_policy.fuse(result, self.specialist.detect(request))

    def _effective_generation(self, requested: GenerationConfig) -> GenerationConfig:
        max_allowed = int(self.config.limits.get("max_output_tokens", 512))
        if requested.max_tokens > max_allowed:
            raise ValueError(f"max_tokens exceeds configured limit {max_allowed}")
        return requested

    def _validate_limits(self, request: ModelRequest) -> None:
        max_pixels = int(self.config.limits.get("max_image_pixels", 16_777_216))
        if request.image_width * request.image_height > max_pixels:
            raise ValueError(f"image exceeds configured pixel limit {max_pixels}")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                candidate = "\n".join(lines[1:-1])
                if candidate.lstrip().startswith("json"):
                    candidate = candidate.lstrip()[4:].lstrip()
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ModelOutputError("model output is not one valid JSON object") from exc
        if not isinstance(payload, dict):
            raise ModelOutputError("model output root must be a JSON object")
        return payload

    def _to_result(
        self,
        payload: dict[str, Any],
        request: ModelRequest,
        raw_text: str,
        metadata: Any,
        generation: GenerationConfig,
    ) -> ModelResult:
        required = {"result", "objects", "reason", "refusal_code"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ModelOutputError(f"model output missing fields: {', '.join(missing)}")
        try:
            decision = Decision(payload["result"])
        except (TypeError, ValueError) as exc:
            raise ModelOutputError("result must be compliant, violation, or uncertain") from exc
        if not isinstance(payload["reason"], str) or not payload["reason"].strip():
            raise ModelOutputError("reason must be a non-empty string")
        raw_objects = payload["objects"]
        if not isinstance(raw_objects, list):
            raise ModelOutputError("objects must be an array")
        objects = tuple(self._parse_object(item, request) for item in raw_objects)

        refusal: RefusalSignal | None = None
        refusal_value = payload["refusal_code"]
        if decision is Decision.UNCERTAIN:
            try:
                code = RefusalCode(refusal_value)
            except (TypeError, ValueError) as exc:
                raise ModelOutputError("uncertain output requires a valid refusal_code") from exc
            refusal = RefusalSignal(code=code, message=payload["reason"].strip())
        elif refusal_value is not None:
            raise ModelOutputError("non-uncertain output must use refusal_code=null")

        if decision is Decision.VIOLATION and not objects:
            raise ModelOutputError("violation output has no evidence object")
        if decision is Decision.COMPLIANT and objects:
            raise ModelOutputError("compliant output contains contradictory positive evidence")
        if decision is Decision.UNCERTAIN and objects:
            raise ModelOutputError("uncertain output must not invent positive evidence")

        provenance = ModelProvenance(
            model_id=self.config.model_id,
            model_revision=self.config.revision,
            backend=self.backend.name,
            config_fingerprint=self.config.fingerprint,
            adapter_id=self.config.adapter_path,
            seed=generation.seed,
            deterministic=generation.deterministic,
            extra=dict(metadata),
        )
        return ModelResult(
            decision=decision,
            objects=objects,
            reason=payload["reason"].strip(),
            uncertain=decision is Decision.UNCERTAIN,
            refusal=refusal,
            provenance=provenance,
            raw_text=raw_text,
        )

    @staticmethod
    def _parse_object(item: Any, request: ModelRequest) -> DetectedObject:
        if not isinstance(item, dict):
            raise ModelOutputError("each object must be a JSON object")
        if not {"label", "bbox", "confidence"}.issubset(item):
            raise ModelOutputError("object requires label, bbox, and confidence")
        bbox = item["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
        ):
            raise ModelOutputError("bbox must contain four numeric values")
        if any(float(value) != int(value) for value in bbox):
            raise ModelOutputError("bbox values must be integer pixel coordinates")
        pixel_bbox = tuple(int(value) for value in bbox)
        x1, y1, x2, y2 = pixel_bbox
        if x2 > request.image_width or y2 > request.image_height:
            raise ModelOutputError("bbox exceeds original image bounds")
        label = item["label"]
        confidence = item["confidence"]
        if not isinstance(label, str):
            raise ModelOutputError("object label must be a string")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ModelOutputError("object confidence must be numeric")
        try:
            return DetectedObject(
                label=label,
                bbox=(x1, y1, x2, y2),
                confidence=float(confidence),
                source=EvidenceSource.VLM,
            )
        except (TypeError, ValueError) as exc:
            raise ModelOutputError(f"invalid evidence object: {exc}") from exc
