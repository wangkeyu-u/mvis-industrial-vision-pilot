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


class AnalysisMode(StrEnum):
    VLM_ONLY = "vlm_only"
    SPECIALIST_ONLY = "specialist_only"
    FUSED = "fused"


class ObjectSource(StrEnum):
    VLM = "vlm"
    RF_DETR = "rf-detr"
    FLORENCE = "florence"
    FUSION = "fusion"
    EFFICIENT_AD = "efficientad"
    PATCHCORE = "patchcore"
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
    analysis_mode: AnalysisMode | None = None
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


class SpecialistOutput(BaseModel):
    """Internal provider-neutral specialist result; raw heatmap bytes are never serialized."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    score: float = Field(ge=0.0)
    threshold: float = Field(ge=0.0)
    source: ObjectSource
    objects: list[EvidenceObject] = Field(default_factory=list)
    heatmap_png: bytes | None = Field(default=None, exclude=True, repr=False)
    warnings: list[str] = Field(default_factory=list, max_length=32)

    @property
    def detected(self) -> bool:
        return self.score >= self.threshold

    @model_validator(mode="after")
    def validate_specialist_evidence(self) -> SpecialistOutput:
        if self.source in {ObjectSource.VLM, ObjectSource.FUSION}:
            raise ValueError("specialist source must identify a specialist implementation")
        if self.detected and not self.objects:
            raise ValueError("positive specialist score requires localized evidence")
        if not self.detected and self.objects:
            raise ValueError("negative specialist score cannot contain localized evidence")
        return self


class HeatmapArtifact(BaseModel):
    artifact_id: str = Field(pattern=r"^hm_[0-9a-f]{32}$")
    uri: str = Field(pattern=r"^/v1/artifacts/heatmaps/hm_[0-9a-f]{32}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: Literal["image/png"] = "image/png"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    expires_in_seconds: int = Field(gt=0)


class SpecialistEvidence(BaseModel):
    specialist_id: str
    revision: str
    score: float = Field(ge=0.0)
    threshold: float = Field(ge=0.0)
    detected: bool
    source: ObjectSource
    objects: list[EvidenceObject]
    heatmap: HeatmapArtifact | None = None
    provenance: ModelProvenance
    quality_status: QualityStatus
    quality_accepted: bool


class ModelIdentity(BaseModel):
    base: str
    adapter: str | None = None
    revision: str = "unversioned"


class QualityStatus(StrEnum):
    UNVALIDATED = "unvalidated"
    PILOT_FAILED = "pilot_failed"
    PILOT_CANDIDATE = "pilot_candidate"
    PILOT_PASSED = "pilot_passed"


class ServingTier(StrEnum):
    PILOT = "pilot"
    PRODUCTION = "production"


class ModelProvenance(BaseModel):
    """Immutable model/config/data/prompt identity published by every API surface."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    checkpoint_revision: str
    backend: str
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{16,64}$")
    config_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    adapter_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    weight_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    data_version: str
    prompt_version: str


class QualityEvidence(BaseModel):
    """Non-secret verification result for a detached evaluator attestation."""

    model_config = ConfigDict(extra="forbid")

    signature_verified: bool = False
    algorithm: str | None = None
    key_id: str | None = None
    report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attestation_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_reason: str | None = None


class LatencyBreakdown(BaseModel):
    preprocess_ms: int = Field(ge=0)
    inference_ms: int = Field(ge=0)
    validation_ms: int = Field(ge=0)


class AnalyzeResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    analysis_mode: AnalysisMode
    model: ModelIdentity
    provenance: ModelProvenance
    quality_status: QualityStatus
    quality_accepted: bool
    serving_tier: ServingTier
    specialist: SpecialistEvidence | None = None
    human_review_required: bool = False
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
    provenance: ModelProvenance | None
    quality_status: QualityStatus | None
    quality_accepted: bool
    serving_tier: ServingTier | None
    runtime: dict[str, Any]


class ModelOpsActionRequest(BaseModel):
    """Compare-and-swap inputs for auditable model lifecycle changes."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=256)
    expected_active_model_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_active_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    serving_tier: ServingTier = ServingTier.PILOT

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
