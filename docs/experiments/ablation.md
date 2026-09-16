# Comparison and ablation boundaries

## Evidence and result
Whole-image post-processing: 0/9; tiled PatchCore: 2/9; supervised U-Net: 6/9 on historical test. This is a route comparison, not a pure architecture ablation: input geometry, supervision and model all change. v2 CV compares tiled PatchCore Acc@IoU=0.0196 to U-Net 0.7451. No new causal gain is inferred from unmatched protocols.

Source: [retained historical report](../model/phase8_algorithm_report.md). These are historical recorded results, not newly rerun training metrics.

## Reproduction

From the repository root, after acquiring the KSDD data, pinned encoder/model assets and required training dependencies described in the source report:

```bash
python -m src.evaluation.phase8_experiments
```

A fresh clone does not include `data/processed` or `artifacts/model`; command presence is not a claim of one-command model reproduction. The current audit runs available code/contract tests separately. [Validation](validation.md).

## Interpretation

Keep internal_pilot_validation, external_holdout=false and production_ready=false. The historical test influenced route selection; the corrected grouped protocol is the primary internal result.
