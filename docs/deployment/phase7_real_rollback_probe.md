# Phase 7 Release Report

- Pilot runtime status: `passed`
- Production release status: `blocked`
- Analysis mode: `specialist_only`
- Runtime measurements used mock: `false`
- Generated: `2026-08-09T00:46:11.707552+00:00`

## Strict real runtime

- Accepted: `true`
- Successful requests: `1/1`
- P50 / P95: `20.143 / 20.143 ms`
- Peak process memory: `4503.14 MB`
- Timeout returned 504: `true`
- Post-timeout recovery: `true`

## Quality and promotion gate

- Quality status: `pilot_failed`
- Signed evaluation accepted: `false`
- Production ready: `false`
- A pilot may be demonstrated when runtime-ready; production remains blocked without an exact-provenance signed evaluator attestation.

## Rollback

- Real candidate rollback exercised: `true`
- Status: `passed`
- Reason: `isolated_registry_rollback_and_http_recovery`

## Evidence limits

- Latency and memory qualify only this strict-real local Uvicorn run and sample.
- Controlled timeout recovery is excluded from latency statistics.
- Mock lifecycle evidence, when present, is control-plane evidence only and never a real-model KPI.
- See the algorithm evaluation package for classification/localization quality and dataset limits.

## Blockers

- `none`
