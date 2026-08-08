"""Public API and model-boundary schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


class AnalyzeTask(StrEnum):
    INSPECT = "inspect"
    GROUND = "ground"
    EXTRACT = "extract"
    VQA = "vqa"


class ObjectSource(StrEnum):
    VLM = "vlm"
    RF_DETR = "rf-detr"
    FLORENCE = "florence"
    FUSION = "fusion"
    MOCK = "mock"


class AnalyzeOptions(BaseModel):
    """Explicit request option allow-list; unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    max_tokens: Annotated[int, Field(ge=1, le=2048)] = 512
    seed: Annotated[int, Field(ge=0, le=2_147_483_647)] = 42


class Base64AnalyzeRequest(BaseModel):
    """JSON transport for clients that cannot send multipart file uploads."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        min_length=1,
        description="A data:image/jpeg|png|webp;base64,... URL",
    )
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)
    query: str = Field(min_length=1, max_length=1000)
    task: AnalyzeTask = AnalyzeTask.INSPECT
    model: str = Field(default="active", min_length=1, max_length=128)
    use_specialist: bool = False
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query cannot be blank")
        return normalized


class ImageMetadata(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: Literal["jpeg", "png", "webp"]
    mode: str
    byte_size: int = Field(gt=0)

    @property
    def log_shape(self) -> list[int]:
        return [self.height, self.width]


class EvidenceObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=256)
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    source: ObjectSource

    @field_validator("bbox")
    @classmethod
    def validate_bbox_order(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = value
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        return value


class ModelOutput(BaseModel):
    """Model-neutral output. Adapters must not leak provider-specific payloads."""

    model_config = ConfigDict(extra="forbid")

    result: str = Field(min_length=1, max_length=512)
    objects: list[EvidenceObject] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=4000)
    uncertain: bool = False
    warnings: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_conclusion_evidence_consistency(self) -> ModelOutput:
        negative_results = {
            "compliant",
            "no_violation",
            "not_found",
            "none",
            "negative",
        }
        if self.result.strip().lower() in negative_results and self.objects:
            raise ValueError("negative conclusion cannot contain evidence objects")
        return self


class ModelIdentity(BaseModel):
    base: str
    adapter: str | None = None


class LatencyBreakdown(BaseModel):
    preprocess_ms: int = Field(ge=0)
    inference_ms: int = Field(ge=0)
    validation_ms: int = Field(ge=0)


class AnalyzeResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    model: ModelIdentity
    result: str
    objects: list[EvidenceObject]
    reason: str
    uncertain: bool
    latency_ms: int = Field(ge=0)
    latency: LatencyBreakdown
    timing: LatencyBreakdown
    warnings: list[str]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["live", "ready", "not_ready"]
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    service: str
    code: str
    api: str
    schema_version: str
    model: ModelIdentity | None
    runtime: dict[str, Any]


class ModelOpsActionRequest(BaseModel):
    """Compare-and-swap inputs for auditable model lifecycle changes."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=256)
    expected_active_model_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_active_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason cannot be blank")
        return normalized


class ModelOpsActionResponse(BaseModel):
    request_id: str
    action: str
    audit: dict[str, Any]
    registry: dict[str, Any]
