# Use supervised U-Net for localization

Recorded: 2026-09-15. This document retrospectively records the rationale behind the current implementation. Code-supported reasoning is an interpretation of the implementation, not a claim that an unrecorded historical experiment took place.

## Context
Whole-image PatchCore retained classification signal while missing all nine historical positive localizations. See [algorithm report](../model/phase8_algorithm_report.md).

## Options Considered

### Option A
Keep whole-image PatchCore. Pros: normal-data training, low latency. Cons: 500×1265→256×256 loses thin-defect geometry; post-processing search still gives 0/9.

### Option B
Use tiled supervised U-Net. Pros: learns pixel masks; historical 6/9. Cons: mask annotation, slower inference and fragmented boxes.

## Decision
Use U-Net as pilot_candidate; retain PatchCore manifests for rollback.

## Why
A classification score cannot repair missing spatial evidence. This is a change in supervision and model, not a free component gain.

## Validation
[Historical repair report](../model/phase8_algorithm_report.md), `src/training/unet_segmentation.py`, `tests/model/test_phase8_service_bridge.py`. Training was not rerun in this audit.

## Trade-offs
Historical test participated in route comparison. The 6/9 result is selection evidence, not independent generalization.

## What Would Change My Mind
An external entity-disjoint dataset where an unsupervised route meets localization, false-positive and latency requirements with less annotation cost.
