# Isolate candidate selection inside each outer fold

Recorded: 2026-09-15. This document retrospectively records the rationale behind the current implementation. Code-supported reasoning is an interpretation of the implementation, not a claim that an unrecorded historical experiment took place.

## Context
v1 selected candidate post-processing configurations using a global validation set whose entities later appeared in outer folds.

## Options Considered

### Option A
Keep global Top-8 candidates. Pros: cheaper search. Cons: indirect selection contamination.

### Option B
Static candidates plus inner-fold selection and entity-cluster bootstrap. Pros: isolates outer entities. Cons: wider uncertainty and more selection work.

## Decision
Use `ksdd_entity_grouped_nested_cv_v2`.

## Why
Entity grouping alone does not eliminate leakage from upstream candidate selection. Related images also invalidate image-independent bootstrap assumptions.

## Validation
`src/evaluation/phase8_1_protocol.py`, `tests/evaluation/test_entity_cv.py`, [methodology correction](../model/phase8_1_methodology_report.md).

## Trade-offs
The dataset still influenced the research direction; internal CV is not external holdout. U-Net uses an inner holdout, not exhaustive nested training.

## What Would Change My Mind
A frozen independent cohort with enough entities to estimate cross-domain uncertainty directly.
