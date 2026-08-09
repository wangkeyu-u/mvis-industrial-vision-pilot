"""Phase 8.1 frozen selection space and leakage guards.

The phase 8 CV runners ranked candidates on the original validation split
before running outer cross-validation.  Those validation entities later
appeared in outer test folds.  This module replaces that contaminated shortlist
with a static, data-independent search space and explicit entity isolation
checks.  Fusion mode, image threshold and post-processing are selected only
inside each outer fold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Iterable, Sequence

from src.evaluation.localization_postprocess import PostprocessConfig

PROTOCOL_VERSION = "ksdd_entity_grouped_nested_cv_v2"
FROZEN_FUSION_MODES = ("max", "mean", "weighted")


def frozen_postprocess_candidates(model_family: str) -> tuple[PostprocessConfig, ...]:
    """Return the protocol-defined grid without consulting any metric artifact."""

    specifications: list[tuple[str, float, int, int, str, int, float]] = []
    if model_family == "unet":
        for threshold in (0.5, 0.7, 0.85):
            for area in (64, 256):
                for morphology in ("none", "dilate3"):
                    specifications.append(
                        ("absolute", threshold, 3, area, morphology, 0, 0.0)
                    )
        for area in (64, 256):
            for morphology in ("none", "dilate3"):
                specifications.append(("absolute", 0.7, 1, area, morphology, 0, 0.0))
        for area in (64, 256):
            specifications.append(("absolute", 0.7, 3, area, "dilate3", 24, 3.0))
    elif model_family == "patchcore":
        for threshold in (0.90, 0.95):
            for components in (3, 0):
                for area in (64, 256):
                    for morphology in ("none", "open3"):
                        specifications.append(
                            ("quantile", threshold, components, area, morphology, 0, 0.0)
                        )
        for threshold in (0.90, 0.95):
            specifications.append(("quantile", threshold, 0, 256, "open3", 24, 3.0))
    else:
        raise ValueError(f"unsupported model family: {model_family}")

    candidates = []
    for mode, threshold, components, area, morphology, gap, aspect in specifications:
        candidates.append(
            PostprocessConfig(
                name=(
                    f"{model_family}_{mode}{threshold}_k{components}_a{area}_"
                    f"{morphology}_g{gap}_ar{aspect}"
                ),
                threshold_mode=mode,
                threshold=threshold,
                max_components=components,
                min_area=area,
                morphology=morphology,
                merge_vertical_gap=gap,
                thin_aspect=aspect,
            )
        )
    return tuple(candidates)


def candidate_grid_fingerprint(
    model_family: str,
    candidates: Sequence[PostprocessConfig],
    fusion_modes: Sequence[str] = FROZEN_FUSION_MODES,
) -> str:
    payload = {
        "model_family": model_family,
        "fusion_modes": list(fusion_modes),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assert_entity_isolation(
    *,
    outer_test_entities: Iterable[str],
    fit_entities: Iterable[str],
    selection_entities: Iterable[str],
    early_stop_entities: Iterable[str] = (),
) -> None:
    """Fail closed if an outer-test entity participates in any selection step."""

    outer = set(outer_test_entities)
    training = set(fit_entities)
    selection = set(selection_entities)
    early_stop = set(early_stop_entities)
    if outer & training:
        raise ValueError("outer-test entity leaked into model training")
    if outer & selection:
        raise ValueError("outer-test entity leaked into threshold/fusion/postprocess selection")
    if outer & early_stop:
        raise ValueError("outer-test entity leaked into early stopping")
