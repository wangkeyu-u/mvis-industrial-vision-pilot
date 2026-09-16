# Supervised repair

## Evidence and result
Historical U-Net test localization 6/9 versus tiled PatchCore 2/9. U-Net P50 147.3 ms versus 28.7 ms tiled and 7.3 ms whole-image. Masks and training are additional supervision; three positive cases still missed.

Source: [retained historical report](../model/phase8_algorithm_report.md). These are historical recorded results, not newly rerun training metrics.

## Reproduction

From the repository root, after acquiring the KSDD data, pinned encoder/model assets and required training dependencies described in the source report:

```bash
python -m src.training.unet_segmentation
```

A fresh clone does not include `data/processed` or `artifacts/model`; command presence is not a claim of one-command model reproduction. The current audit runs available code/contract tests separately. [Validation](validation.md).

## Interpretation

Keep internal_pilot_validation, external_holdout=false and production_ready=false. The historical test influenced route selection; the corrected grouped protocol is the primary internal result.
