# Phase 6 Quality Gate and Adapter Release Report

## Outcome

The backend now reports an immutable model provenance snapshot on `/version`,
`/v1/models`, `/v1/analyze`, and structured request logs. The previously missing
revision is sourced from the pinned model configuration and is never synthesized
as `not_reported`.

Runtime readiness and quality acceptance are independent:

- `runtime_ready` means the active adapter is loaded and callable.
- `quality_status` is one of `unvalidated`, `pilot_failed`, `pilot_candidate`, or
  `pilot_passed`.
- `quality_accepted=true` requires both `pilot_passed` and a valid evaluator
  attestation whose signature and provenance match.
- `production_ready=true` additionally requires an active `production` serving
  tier. A runtime-ready pilot can therefore remain healthy without being
  represented as production-qualified.

The current zero-shot Qwen entry is explicitly `pilot_failed`, based on the
phase-5 real probe. It may be served only with `serving_tier=pilot`; every analyze
response includes `quality:pilot_failed` and `serving_tier:pilot` warnings. The
LoRA registry entry is `unvalidated` and not runtime-ready until a local adapter
artifact exists. Neither condition is silently promoted.

## Published provenance

The default service config pins:

- model: `mlx-community/Qwen3-VL-2B-Instruct-4bit`
- checkpoint revision: `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`
- base weight SHA-256:
  `4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb`
- zero-shot evaluation data version: `ksdd-0.1.0`
- LoRA training data version: `ksdd_sft-1.0.0`
- prompt version: `ksdd_prompt_v1`
- adapter hash: `null` for zero-shot; deterministic SHA-256 of the complete local
  adapter tree for LoRA.

The registry exposes named `zero_shot` and `lora` aliases in real/auto builds,
in addition to lifecycle aliases (`active`, `candidate`, `previous`). Adapter
paths and evaluator secrets are not returned.

## Evaluator attestation contract

Production activation is closed unless the service receives both environment
variables:

```bash
export MVIS_EVALUATOR_ATTESTATION=/secure/release/evaluator-attestation.json
export MVIS_EVALUATOR_SIGNING_KEY='at-least-32-bytes-from-a-secret-store'
```

The JSON contract is
`configs/service/evaluator_attestation.schema.json`. The signature value is
lowercase HMAC-SHA256 over `payload` encoded as UTF-8 JSON with sorted keys and
separators `(',', ':')`; the signing key is never read from YAML. The signed
payload binds model ID, checkpoint revision, adapter hash, data version, prompt
version, evaluator identity, report hash, and the acceptance decision. A bad
signature, short/missing key, malformed report, or any provenance mismatch leaves
the model unaccepted.

The evaluator's current KSDD test split is pilot-only (56 samples, below the 300
sample formal floor). Such a package must not set
`eligible_for_model_acceptance=true`; consequently it cannot produce a valid
production attestation even if exploratory metrics pass.

## Atomic switch and rollback

Pilot activation is explicit:

```bash
curl -s -X POST http://127.0.0.1:8001/v1/models/qwen3-vl-2b-instruct-4bit-lora-r8/activate \
  -H "X-ModelOps-Token: $MVIS_MODELOPS_TOKEN" \
  -H 'X-ModelOps-Actor: release-operator' \
  -H 'Content-Type: application/json' \
  -d '{
    "reason":"controlled pilot",
    "serving_tier":"pilot",
    "expected_active_model_id":"qwen3-vl-2b-instruct-4bit"
  }'
```

For production, use `"serving_tier":"production"`; the same compare-and-swap
operation fails with `MODELOPS_CONFLICT` unless the target has a matching signed
attestation. Failed activation leaves all states and aliases unchanged. Rollback
accepts the same serving tier and gate, atomically moves `previous` to `active`,
and retires the rolled-back model.

## Candidate probe

No model is downloaded by this command. Before the algorithm artifact exists it
returns machine-readable `status=blocked`, `mock_used=false`, and
`lora_artifact_unavailable`:

```bash
MVIS_MODEL_MODE=real PYTHONPATH=. .venv/bin/python -m src.api candidate-probe
```

Attach the algorithm's bounded training diagnostics when training did not emit
weights:

```bash
MVIS_MODEL_MODE=real PYTHONPATH=. .venv/bin/python -m src.api candidate-probe \
  --run-manifest artifacts/model/phase6/qlora_smoke_16_retry1/run_manifest.json
```

After the algorithm thread publishes the local adapter:

```bash
MVIS_MODEL_MODE=real \
MVIS_LORA_ADAPTER_PATH=/absolute/local/path/to/adapter \
PYTHONPATH=. .venv/bin/python -m src.api candidate-probe \
  --sample data/processed/ksdd_v0/probes/ksdd_kos10_part3.jpg
```

The probe hashes the complete adapter tree without following symlinks, loads only
that candidate, validates and activates it as an isolated pilot, optionally runs
one real analyze request, and prints provenance, response, duration, and process
peak memory as JSON. It never falls back to Mock and never enables downloads.

## Verification evidence

The algorithm thread has completed the 56-image zero-shot baseline. Read-only
inspection of its evaluator output shows `macro_f1=0.0`, `acc_at_iou=0.0`, and
`json_schema_validity_rate=0.9821428571`; all 56 samples are failure cases. This
confirms `pilot_failed`, and the pilot-sized dataset remains ineligible for a
production signature. The 16-step LoRA smoke ended as
`resource_limit_exceeded`: MLX recorded `15997.066 MB` peak against the
`12288 MB` limit, Metal reported insufficient memory, and adapter path/hash are
null. Consequently there is no real candidate to load. The strict probe reports
both `lora_artifact_unavailable` and `training_resource_limit_exceeded`, plus the
training manifest SHA-256
`98773386d1c62552a768744ed4fe753987e4f1f3e3993d857f881a53e4ebbba0`, instead
of fabricating a load success. Completed
backend verification:

- API/security/ModelOps tests: `74 passed`.
- algorithm service bridge, Qwen adapter, and diagnostics contract selection:
  `16 passed`.
- Ruff over backend-owned source and tests: passed.

A final strict-real Uvicorn smoke used a frozen KSDD probe after the training
process exited. `/health/ready`, `/version`, `/v1/models`, and `/v1/analyze` all
returned the pinned revision and the same provenance. The active zero-shot entry
was runtime-ready but remained `pilot_failed`, `quality_accepted=false`,
`serving_tier=pilot`, and `production_ready=false`; the LoRA entry was present as
an unready `unvalidated` candidate with `adapter_hash=null`. The analyze call took
1,975 ms service time and process historical peak reached 2,078.97 MB. This is a
single integration smoke, not a performance KPI or quality score.

The zero-shot/LoRA quality comparison and real candidate load result must be
appended only after the algorithm and evaluator artifacts exist. Mock latency is
not a model KPI and is not used here.
