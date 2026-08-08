# Model Card Template

> Template state: `not_run`. Replace a field only after recording the supporting artifact. Use `unavailable` when the evidence cannot be obtained; never substitute an estimate.

## Identity

| Field | Value | Evidence status |
|---|---|---|
| Model ID | `unavailable` | `unavailable` |
| Immutable revision | `unavailable` | `unavailable` |
| Base-weight hash | `unavailable` | `unavailable` |
| Adapter ID/hash | `unavailable` | `not_run` |
| Quantization | `unavailable` | `unavailable` |
| License review | `unavailable` | `unavailable` |
| Intended hardware | M5 MacBook Air / 16GB | `recorded` |
| Intended domain | visual compliance review | `recorded` |

## Intended use and exclusions

Intended use: assist human review with structured decisions, localized visual evidence, explicit refusal signals, and source provenance.

Excluded use: autonomous punishment, takedown, medical, law-enforcement, credit, or other high-impact decisions without qualified human review.

## Evaluation evidence

| Measurement | Value | Evidence status | Artifact |
|---|---|---|---|
| Macro-F1 | `unavailable` | `not_run` | `unavailable` |
| Acc@IoU 0.5 | `unavailable` | `not_run` | `unavailable` |
| JSON schema validity | `unavailable` | `not_run` | `unavailable` |
| Hard-negative false-positive rate | `unavailable` | `not_run` | `unavailable` |
| Evidence/conclusion consistency | `unavailable` | `not_run` | `unavailable` |
| Refusal rate by slice | `unavailable` | `not_run` | `unavailable` |

## Runtime evidence

| Measurement | Value | Evidence status |
|---|---|---|
| Warm-up runs | `unavailable` | `not_run` |
| P50 latency | `unavailable` | `not_run` |
| P95 latency | `unavailable` | `not_run` |
| Peak memory | `unavailable` | `not_run` |
| 100-request stability | `unavailable` | `not_run` |

## Training and adaptation

Training/adaptation status: `not_run`.

- Frozen data version: `unavailable`
- Experiment spec fingerprint: `unavailable`
- Seed: `unavailable`
- LoRA/QLoRA configuration: `unavailable`
- Checkpoint selection rule: `unavailable`
- Training run manifest: `unavailable`

## Limitations and risk controls

- Real-model accuracy and hardware performance remain `not_run` until backed by a run manifest and artifacts.
- Low-confidence, conflicting, or insufficient evidence must produce an explicit refusal and human-review signal.
- Model and specialist provenance must remain attached to each prediction.
- Offline configuration must not silently download or change model revisions.

## Approval

| Field | Value | Evidence status |
|---|---|---|
| Reviewer | `unavailable` | `not_run` |
| Review date | `unavailable` | `not_run` |
| Release decision | `unavailable` | `not_run` |
| Evidence bundle | `unavailable` | `not_run` |
