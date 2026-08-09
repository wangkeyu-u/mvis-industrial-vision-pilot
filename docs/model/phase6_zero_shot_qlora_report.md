# Phase 6 — KSDD V0 real zero-shot baseline and controlled QLoRA

## Outcome

The complete frozen 56-image KSDD V0 test split was evaluated with the real cached
Qwen3-VL-2B 4-bit model. The run is reproducible but is not a usable quality baseline:
55/56 outputs were structurally valid, yet every valid output abstained as `uncertain`, so
classification macro-F1 and localization accuracy were both 0.

The single rank-8 QLoRA candidate was attempted exactly as a controlled 16-micro-step smoke.
It failed during the first training step with a Metal insufficient-memory error. MLX reported
a 15,997.066 MiB peak, above the 12,288 MiB gate. No adapter was written. Consequently the
100-step formal run and candidate evaluation are explicitly `not_run`, and the fair comparison
is `unavailable`; no failed run is represented as a candidate result.

All results are pilot evidence. The frozen test population has 56 samples, below the formal KPI
floor of 300 recorded by the dataset manifest.

## Frozen identities

| Item | Frozen value |
| --- | --- |
| Dataset | `ksdd-0.1.0`, frozen test 56 (47 hard negatives, 9 positives) |
| Dataset manifest SHA-256 | `fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a` |
| Prompt | `ksdd_prompt_v1` |
| Prompt query SHA-256 | `cba1c2914ce8d7e2eff0abb2ad188fe1103f34919846b49fa9a6b191754d28fc` |
| Prompt spec SHA-256 | `b2bcae976118a00a919fee2224b44259892cec480494bb1e0c4c8f1f86ebe1af` |
| Model | `mlx-community/Qwen3-VL-2B-Instruct-4bit` |
| Revision | `9c4f5209e57b31f4b9dfba735de3fb983739c9cc` |
| Weight SHA-256 | `4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb` |
| Generation | seed `20260808`, greedy, temperature `0`, top-p `1`, max tokens `256` |

## Zero-shot results

| Metric | Real result |
| --- | ---: |
| Complete test inputs attempted | 56 / 56 |
| Evaluator-compatible valid outputs | 55 / 56 (98.214%) |
| Valid prediction distribution | 55 uncertain, 0 compliant, 0 violation |
| Macro-F1 | 0.000 |
| Classification correct | 0 / 56 |
| Localization Acc@IoU 0.5 | 0 / 9 (0.000) |
| Hard-negative false-positive rate | 0 / 47 (0.000) |
| Evidence/conclusion consistency | 55 / 56 (98.214%) |
| Failure cases | 56 / 56 |

The hard-negative false-positive rate of zero is not evidence of useful specificity: the model
abstained on all 47 hard negatives, giving zero decisive coverage. All nine positives were also
abstentions and localization misses. The only invalid output, `ksdd_kos43_part1`, returned
`result=uncertain` but invented the unsupported refusal code `uncertain`; the adapter correctly
rejected it rather than coercing it.

The prompt asks for label `defect`, while frozen localization truth uses `surface_defect`. This
did not change this run's localization score because the model emitted no boxes, but it is a
known label-ontology risk for future experiments and must be resolved before a new frozen prompt
version is created.

## Real inference performance

| Measurement | Result |
| --- | ---: |
| Model load | 1,768.149 ms |
| Per-image P50 | 1,882.164 ms |
| Per-image P95 | 2,307.581 ms |
| Minimum / maximum | 1,676.247 / 2,425.169 ms |
| MLX allocator peak | 2,763.019 MiB |
| Process peak RSS | 706.781 MiB |

MLX allocator peak and process RSS are recorded separately; neither alone represents system-wide
unified-memory pressure. Predictions SHA-256 is
`fef7c67de3457a727d1d213621d263eeecfb5c1aa4b01761bd1adf694139c801`.

## SFT package and controlled training

The data-thread exporter was invoked without changing its implementation. Its self-contained
runtime package is under `artifacts/model/phase6/ksdd_sft_v1`:

- train 271, validation 56, test 56;
- strict answer JSON validity 100%;
- train/test sample, entity, and image-SHA overlap all empty;
- SFT manifest SHA-256
  `250383fe99f36eda6299f47d5223a8361ee89198d44d3d5376523a83726e5f23`;
- train SHA-256 `4e5d9528ddc3608c2a3300fb27c1ec133717caa247eda767c6259d49807c4017`;
- validation SHA-256 `6453453ae5ba21de8c39ac1dda7da3e84c157c38021cc8fa27020ed49f588ab1`.

The installed `mlx-vlm==0.6.10` training extra supplied `datasets==5.0.1`. The controlled runner
uses the official `VisionDataset`, `setup_model_for_training`, `TrainingArgs`, and `train`
interfaces. It passes the real validation dataset, unlike the installed generic CLI which loads
only one split and passes `val_dataset=None`. The source SFT records remain unchanged; an
in-memory, named `model_adapter_refusal_code_v1` transform adds `refusal_code=null` to
non-uncertain assistant JSON so generated answers match the ModelAdapter contract.

Requested smoke configuration:

| Setting | Value |
| --- | ---: |
| Method | QLoRA over 4-bit base, language modules only |
| Rank / alpha / dropout | 8 / 16 / 0.05 |
| Batch / accumulation / effective batch | 1 / 16 / 16 |
| Requested steps | 16 |
| Fixed seed | 20260808 |
| Learning rate | 1e-4 |
| Max sequence length | 768 |
| Gradient checkpointing | enabled |
| Completion-only loss | enabled |
| Validation batches | 2 |

The data loaded and LoRA setup succeeded: 8.716288M of 2,127.532032M parameters were trainable
(0.410%). Two validation batches completed before step 1. The first training step then failed
with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Wall time was 324.516 seconds; MLX peak was
15,997.066 MiB. The independent `vmmap` observation reached a 16.0 GiB physical-footprint peak.
No training step was confirmed complete and no adapter/config was saved.

The first smoke attempt is also retained: it failed before training because Datasets 5.0 passed
a non-serializable `LazyRow`. The runner was fixed to materialize the Mapping, tested, and rerun;
the retained failure is not used for the resource decision.

Official interface references: [MLX-VLM LoRA guide](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/LORA.MD)
and [MLX-VLM repository](https://github.com/Blaizzy/mlx-vlm).

## Commands and verification

```bash
.venv/bin/python -m src.inference.ksdd_baseline \
  --spec configs/models/ksdd_zero_shot_prompt_v1.json \
  --manifest data/processed/ksdd_v0/manifest.json \
  --samples data/processed/ksdd_v0/samples.jsonl \
  --output-dir artifacts/model/phase6/zero_shot

.venv/bin/python -m src.evaluation.cli \
  --ground-truth data/processed/ksdd_v0/evaluation_ground_truth.jsonl \
  --predictions artifacts/model/phase6/zero_shot/predictions.jsonl \
  --output artifacts/model/phase6/zero_shot/evaluation_report.json

.venv/bin/python -m src.training.mlx_vlm_runner \
  --training-config configs/models/ksdd_qwen3_vl_2b_qlora_phase6.json \
  --model-config configs/models/qwen3_vl_2b_mlx_4bit.json \
  --sft-root artifacts/model/phase6/ksdd_sft_v1 \
  --output-dir artifacts/model/phase6/qlora_smoke_16_retry1 \
  --steps 16 --run-kind smoke --validation-batches 2

.venv/bin/python -m pytest tests/model -q
```

Key evidence:

- zero-shot run: `artifacts/model/phase6/zero_shot/run_manifest.json`;
- predictions/raw output/evaluator package: `artifacts/model/phase6/zero_shot/`;
- resource failure: `artifacts/model/phase6/qlora_smoke_16_retry1/run_manifest.json` and
  `training.log`;
- explicit unavailable comparison: `artifacts/model/phase6/comparison_status.json`.

## Next controlled experiment plan

No additional candidate is run in this phase. A later experiment must be separately approved and
versioned. The first resource-reduction candidate should keep rank 8, batch 1, seed, prompt, and
test split fixed while bounding vision tokens through a documented training-only resize and using
smaller gradient accumulation. It must first show both a completed 10–20-step smoke and peak
unified memory below 12 GiB before any formal run. A candidate adapter, if produced, must then be
evaluated on the identical 56 IDs with the same `prompt_v1` generation settings; comparison must
remain pilot-only and must report regressions or no improvement unchanged.
