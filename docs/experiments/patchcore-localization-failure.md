# Localization failure

## Evidence and result
Five classification true positives had low-overlap/offset boxes (IoU 0.117–0.350); four positives missed classification. Whole-image post-processing search (1152 candidates) retained 0/9. Tiling improved to 2/9, still inadequate.

Source: [retained historical report](../qa/phase8_localization_failure_report.md). These are historical recorded results, not newly rerun training metrics.

## Reproduction

From the repository root, after acquiring the KSDD data, pinned encoder/model assets and required training dependencies described in the source report:

```bash
python -m src.evaluation.phase8_failure_analysis
```

A fresh clone does not include `data/processed` or `artifacts/model`; command presence is not a claim of one-command model reproduction. The current audit runs available code/contract tests separately. [Validation](validation.md).

## Interpretation

Keep internal_pilot_validation, external_holdout=false and production_ready=false. The historical test influenced route selection; the corrected grouped protocol is the primary internal result.
