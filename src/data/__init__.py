"""Dataset contracts and quality tooling for visual compliance data."""

from .data_card import generate_data_card
from .deduplication import (
    DuplicateGroup,
    cross_split_leakage_rate,
    find_near_duplicate_groups,
    hamming_distance,
    perceptual_hash,
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
    "LeakageReport",
    "ManifestEntry",
    "ObjectAnnotation",
    "SampleResponse",
    "SchemaError",
    "SplitRatios",
    "cross_split_leakage_rate",
    "build_manifest",
    "detect_split_leakage",
    "compute_dataset_statistics",
    "entity_isolated_split",
    "find_near_duplicate_groups",
    "hamming_distance",
    "generate_data_card",
    "load_samples_jsonl",
    "perceptual_hash",
    "prepare_dataset",
    "require_allowed_license",
    "validate_dataset",
    "validate_image",
]
