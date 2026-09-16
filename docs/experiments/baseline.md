# PatchCore baseline

## Evidence and result
Whole-image PatchCore historical Macro-F1=0.78125, localization Acc@IoU=0/9. The same model can classify anomaly presence while failing spatial overlap.

Source: [retained historical report](../model/phase8_algorithm_report.md). These are historical recorded results, not newly rerun training metrics.

## Reproduction

From the repository root, after acquiring the KSDD data, pinned encoder/model assets and required training dependencies described in the source report:

```bash
python -m src.inference.patchcore_reinfer
```

A fresh clone does not include `data/processed` or `artifacts/model`; command presence is not a claim of one-command model reproduction. The current audit runs available code/contract tests separately. [Validation](validation.md).

## Interpretation

Keep internal_pilot_validation, external_holdout=false and production_ready=false. The historical test influenced route selection; the corrected grouped protocol is the primary internal result.
