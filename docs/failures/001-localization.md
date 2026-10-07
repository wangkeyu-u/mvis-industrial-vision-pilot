# Classification success did not imply localization success

## Symptom
Historical PatchCore Acc@IoU=0/9 while classification Macro-F1=0.78125.

## Reproduction
Use [localization experiment](../experiments/README.md#localization-failure); raw historical artifacts/data are external to Git.

## Root Cause
Reported image resizing destroyed elongated scratch geometry; heatmaps concentrated on partial defects.

## Attempts
Whole-image post-processing 1152 candidates failed; tiled route improved to 2/9.

## Final Fix
Partial repair: supervised U-Net historical 6/9, with corrected CV separately documented. This was not repaired by changing evaluation labels.

## Remaining Risk
Annotation cost; three missed positives; fragmented masks; no external holdout. See retained reports for exact cases.
