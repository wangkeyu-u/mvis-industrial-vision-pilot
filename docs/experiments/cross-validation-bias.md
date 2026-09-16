# CV selection bias correction

## Evidence and result
Static 18-candidate sets and inner-fold fusion selection replace global preselection. Current reported U-Net Acc@IoU=0.7451, entity-cluster CI [0.6226,0.8627], Pixel Dice=0.2337, Box F1=0.3016; retain lower v2 results.

Source: [retained historical report](../model/phase8_1_methodology_report.md). These are historical recorded results, not newly rerun training metrics.

## Reproduction

From the repository root, after acquiring the KSDD data, pinned encoder/model assets and required training dependencies described in the source report:

```bash
python -m src.evaluation.phase8_cv_unet
```

A fresh clone does not include `data/processed` or `artifacts/model`; command presence is not a claim of one-command model reproduction. The current audit runs available code/contract tests separately. [Validation](validation.md).

## Interpretation

Keep internal_pilot_validation, external_holdout=false and production_ready=false. The historical test influenced route selection; the corrected grouped protocol is the primary internal result.
