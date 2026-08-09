"""Dataset contracts and quality tooling for visual compliance data."""

from .data_card import generate_data_card
from .deduplication import (
    DuplicateGroup,
    cross_split_leakage_rate,
    find_near_duplicate_groups,
    hamming_distance,
    perceptual_hash,
)
from .ksdd import (
    KSDD_ARCHIVE_URL,
    KSDD_LICENSE,
    KSDD_SOURCE_PAGE,
    KSDD_V0_ARCHIVE_BYTES,
    KSDD_V0_ARCHIVE_SHA256,
    KSDDPreparationResult,
    KSDDSourceRecord,
    discover_ksdd,
    mask_bounding_box,
    prepare_ksdd_dataset,
)
from .ksdd_sft import (
    DEFAULT_DATASET_VERSION as KSDD_SFT_DATASET_VERSION,
)
from .ksdd_sft import (
    DEFAULT_PROMPT_VERSION as KSDD_SFT_PROMPT_VERSION,
)
from .ksdd_sft import (
    KSDDSFTExportResult,
    build_sft_answer,
    build_sft_record,
    export_ksdd_sft,
    load_sft_jsonl,
    prompt_variant_for,
    validate_sft_record,
)
from .leakage import DuplicateLeak, LeakageReport, detect_split_leakage
from .license_policy import LicenseDecision, LicensePolicy, require_allowed_license
from .manifest import DatasetManifest, ManifestEntry, build_manifest
from .pipeline import DatasetPreparationResult, load_samples_jsonl, prepare_dataset
from .schema import (
    BoundingBox,
    DataSample,
    LicenseInfo,
    ObjectAnnotation,
    SampleResponse,
    SchemaError,
)
from .splitting import SplitRatios, entity_isolated_split
from .statistics import compute_dataset_statistics
from .validation import DatasetValidationReport, ImageMetadata, validate_dataset, validate_image

__all__ = [
    "BoundingBox",
    "DataSample",
    "DatasetManifest",
    "DatasetPreparationResult",
    "DatasetValidationReport",
    "DuplicateGroup",
    "DuplicateLeak",
    "ImageMetadata",
    "LicenseInfo",
    "LicenseDecision",
    "LicensePolicy",
    "KSDD_ARCHIVE_URL",
    "KSDD_LICENSE",
    "KSDD_SOURCE_PAGE",
    "KSDD_V0_ARCHIVE_BYTES",
    "KSDD_V0_ARCHIVE_SHA256",
    "KSDDPreparationResult",
    "KSDDSourceRecord",
    "KSDDSFTExportResult",
    "KSDD_SFT_DATASET_VERSION",
    "KSDD_SFT_PROMPT_VERSION",
    "LeakageReport",
    "ManifestEntry",
    "ObjectAnnotation",
    "SampleResponse",
    "SchemaError",
    "SplitRatios",
    "cross_split_leakage_rate",
    "build_manifest",
    "build_sft_answer",
    "build_sft_record",
    "detect_split_leakage",
    "compute_dataset_statistics",
    "entity_isolated_split",
    "export_ksdd_sft",
    "discover_ksdd",
    "find_near_duplicate_groups",
    "hamming_distance",
    "generate_data_card",
    "load_samples_jsonl",
    "load_sft_jsonl",
    "mask_bounding_box",
    "perceptual_hash",
    "prepare_dataset",
    "prepare_ksdd_dataset",
    "prompt_variant_for",
    "require_allowed_license",
    "validate_dataset",
    "validate_image",
    "validate_sft_record",
]
