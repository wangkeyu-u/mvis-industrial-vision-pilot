# Outer-fold grouping did not prevent global candidate leakage

## Symptom
v1 Pixel Dice=0.4525 fell to 0.2337 after selection correction.

## Reproduction
Inspect `load_cv_candidates` history and [v2 protocol](../evaluation/phase8_1_entity_grouped_protocol.md); run `uv run pytest tests/evaluation/test_entity_cv.py -q`.

## Root Cause
Global validation-based candidate preselection leaked outer-entity information; image bootstrap ignored entity clustering.

## Attempts
Retained v1 report, then switched to static candidates/inner selection/cluster bootstrap.

## Final Fix
Current protocol asserts disjoint selection entities and records candidate provenance.

## Remaining Risk
Still internal evidence; the dataset influenced route decisions. No new independent holdout was acquired.
