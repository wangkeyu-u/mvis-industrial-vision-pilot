# Model experiment records

These are recorded historical model results, not training rerun during repository
cleanup. Full training requires KSDD data, pinned encoders, weights and manifests
that are not included in Git. See [software validation](validation.md) for the
separate code/contract test scope.

All results remain `internal_pilot_validation`, `external_holdout=false` and
`production_ready=false`. The historical test influenced route selection; the
corrected grouped protocol is the primary internal result.


## Baseline

Whole-image PatchCore historical Macro-F1=0.78125, localization Acc@IoU=0/9. The same model can classify anomaly presence while failing spatial overlap.

Source: [retained historical report](../model/phase8_algorithm_report.md). These are historical recorded results, not newly rerun training metrics.

After supplying the assets and dependencies described in the source report:

```bash
python -m src.inference.patchcore_reinfer
```


## Localization failure

Five classification true positives had low-overlap/offset boxes (IoU 0.117–0.350); four positives missed classification. Whole-image post-processing search (1152 candidates) retained 0/9. Tiling improved to 2/9, still inadequate.

Source: [retained historical report](../qa/phase8_localization_failure_report.md). These are historical recorded results, not newly rerun training metrics.

After supplying the assets and dependencies described in the source report:

```bash
python -m src.evaluation.phase8_failure_analysis
```


## U-Net repair

Historical U-Net test localization 6/9 versus tiled PatchCore 2/9. U-Net P50 147.3 ms versus 28.7 ms tiled and 7.3 ms whole-image. Masks and training are additional supervision; three positive cases still missed.

Source: [retained historical report](../model/phase8_algorithm_report.md). These are historical recorded results, not newly rerun training metrics.

After supplying the assets and dependencies described in the source report:

```bash
python -m src.training.unet_segmentation
```


## CV correction

Static 18-candidate sets and inner-fold fusion selection replace global preselection. Current reported U-Net Acc@IoU=0.7451, entity-cluster CI [0.6226,0.8627], Pixel Dice=0.2337, Box F1=0.3016; retain lower v2 results.

Source: [retained historical report](../model/phase8_1_methodology_report.md). These are historical recorded results, not newly rerun training metrics.

After supplying the assets and dependencies described in the source report:

```bash
python -m src.evaluation.phase8_cv_unet
```


## Route comparison

Whole-image post-processing: 0/9; tiled PatchCore: 2/9; supervised U-Net: 6/9 on historical test. This is a route comparison, not a pure architecture ablation: input geometry, supervision and model all change. v2 CV compares tiled PatchCore Acc@IoU=0.0196 to U-Net 0.7451. No new causal gain is inferred from unmatched protocols.

Source: [retained historical report](../model/phase8_algorithm_report.md). These are historical recorded results, not newly rerun training metrics.

After supplying the assets and dependencies described in the source report:

```bash
python -m src.evaluation.phase8_experiments
```
