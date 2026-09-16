# Separate pilot serving from production acceptance

Recorded: 2026-09-15. This document retrospectively records the rationale behind the current implementation. Code-supported reasoning is an interpretation of the implementation, not a claim that an unrecorded historical experiment took place.

## Context
A runnable API and a loadable model are insufficient evidence of accepted model quality.

## Options Considered

### Option A
Treat readiness or mock tests as release approval. Pros: simple deployment. Cons: accepts unmeasured quality.

### Option B
Bind evaluation provenance and signed attestation to acceptance. Pros: fail-closed deployment boundary. Cons: evaluator/key operations must be managed.

## Decision
Keep `production_ready=false` and require bound quality evidence for production activation.

## Why
Model/config/data/prompt identity must correspond to the evaluated system.

## Validation
`src` service acceptance/manifest code and `tests/model/test_phase8_service_bridge.py`; [deployment guide](../deployment/README.md).

## Trade-offs
Local pilot success does not certify real factory performance; a signing mechanism does not create a valid dataset.

## What Would Change My Mind
Independent acceptance data and valid bound evaluator attestation, rather than a better headline metric.
