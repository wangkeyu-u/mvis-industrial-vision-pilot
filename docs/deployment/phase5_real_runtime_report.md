# Phase 5 Real Runtime Report

- Status: `passed`
- Requested runtime: `real`
- Real ready: `true`
- Mock used as real evidence: `false`
- Generated: `2026-08-08T03:42:26.349188+00:00`

## Hardware and model

- Hardware: `Apple M5` / `16384 MB`
- OS: `Darwin 25.5.0`
- Python: `3.13.13`
- Model: `mlx-community/Qwen3-VL-2B-Instruct-4bit`
- Base model: `Qwen/Qwen3-VL-2B-Instruct`
- Revision: `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`
- Config fingerprint: `71b87d18c01aeb4d`
- Weight SHA-256: `4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb`
- Quantization: `4-bit/affine`

## Sample and protocol

- Sample: `system_architecture.png`
- Sample SHA-256: `aef0bd7e638f932dd57fd2701fe0b57dae689b595cbd6ac605c64d38d4530343`
- Dimensions: `750x380`
- Query: `找出不符合要求的区域，并给出可核验的证据框。`
- Warm-up probes: `5`
- Measured continuous probes: `30`
- Timeout recovery: controlled one-shot adapter timeout followed by a real adapter request; excluded from latency KPI.

## Runtime qualification

- Performance eligible: `true`
- Qualification: `real_model_runtime`
- Successful measured runs: `30`
- P50 latency: `872.967 ms`
- P95 latency: `1543.057 ms`
- Peak process memory: `2047.11 MB`
- Memory budget: `12288 MB`

## Observed service output

- HTTP status: `200`
- Model identity: `qwen3-vl-2b-instruct-4bit`
- Result: `compliant`
- Uncertain: `False`
- Evidence objects: `0`
- Reason: 图像中未发现任何不符合要求的区域。所有元素均符合要求，且无证据框。

## Quality scope

- Quality eligible from this runtime protocol: `false`
- Status: `failed_external_algorithm_probe`
- External evidence: `docs/model/phase5_real_model_report.md`
- Runtime latency and memory qualification does not imply visual-compliance quality acceptance.

## Runtime blockers

- None

## Interpretation

Real-model metrics are qualified only after strict real readiness, Uvicorn HTTP probes, 30 successful measured requests, timeout recovery, and the configured memory budget all pass. Mock measurements are never substituted.
