# Phase 7 — KSDD tiled specialist dataset report

## Outcome

KSDD Tiled Specialist V1 is frozen as a deterministic derivative of the KSDD V0
physical-entity split. It contains 256 px and 384 px crops for MLX-VLM SFT and an
Anomalib-compatible folder/sample-manifest export. No model, training job, or inference job was
run. All reported counts below are dataset construction facts, not model scores.

- Dataset version: `ksdd_tiled-1.0.0`
- Source manifest SHA-256: `fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a`
- Tile manifest SHA-256: `7fed681c0a6fef824bf8dbd53948e12d43733cf84088e3fbb21fae3208cb76c5`
- Tile records SHA-256: `9a1f6096ba0e9409d7120f0b018b633eca63aab37bf0dd83702b1491808e6dc6`
- Leakage report SHA-256: `bf765f8eafcfa019bfa40d7aae188737aa57aa38294095b4ad0a8688fcd5ce28`
- Status: `pilot_only`

The source is the official [Kolektor Surface-Defect Dataset release](https://www.vicos.si/resources/kolektorsdd/),
under CC BY-NC-SA 4.0. This derivative remains research/non-commercial and share-alike. The
Anomalib layout and `samples.csv` fields follow its official custom-dataset contract:
`image_path`, `split`, `label_index`, and `mask_path` ([official dataset guide](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/datasets.html)).

## Construction contract

Every tile inherits the frozen source `entity_id` and split. There is no re-splitting: train,
validation, and test source images only produce tiles in the same split. No test source image,
entity, or tile is used for training.

For a defect image, deterministic windows jointly cover every non-zero mask pixel. Each positive
tile must contain mask pixels; its local bbox is recomputed from the cropped binary mask. After
translation back to original coordinates, the union of those local boxes must exactly reconstruct
the source bbox. A defect image also contributes one deterministic, mask-free same-image negative
per tile size. Every normal image contributes one deterministic normal crop per size.

The saved transform is lossless and explicit:

```text
x_original = x_tile + offset_x
y_original = y_tile + offset_y
```

For a lower-resolution heatmap, bbox coordinates are first scaled to tile resolution and then
translated. All boxes use exclusive `xyxy` coordinates.

## Frozen statistics

| Tile size | Split | Positive | Negative | Total | Positive ratio |
|---:|---|---:|---:|---:|---:|
| 256 | train | 53 | 271 | 324 | 16.36% |
| 256 | validation | 12 | 56 | 68 | 17.65% |
| 256 | test | 15 | 56 | 71 | 21.13% |
| 384 | train | 37 | 271 | 308 | 12.01% |
| 384 | validation | 7 | 56 | 63 | 11.11% |
| 384 | test | 11 | 56 | 67 | 16.42% |
| **All** | **all** | **135** | **766** | **901** | **14.98%** |

Negative provenance is 666 normal-image tiles and 100 same-image non-defect-region tiles. Minimum
source-mask coverage is 100%; source-bbox reconstruction is 100%.

Leakage gates all pass: train/test entity overlap 0, train/test source-sample overlap 0,
train/test exact tile SHA-256 overlap 0, and test-derived training tiles 0.

## Export layout

```text
data/processed/ksdd_tiled_v1/
├── tiles/{256,384}/{images,masks}/
├── mlx_vlm/{256,384}/hf/{train,valid,test}.jsonl
├── anomalib/{256,384}/{train,validation,test}/{good,defect,mask/defect}/
├── anomalib/{256,384}/samples.{csv,jsonl}
├── tile_records.jsonl
├── leakage_report.json
├── DATA_CARD.md
└── tile_manifest.{json,sha256}
```

The MLX-VLM assistant answer is strict JSON. Positive answers contain a tile-local
`surface_defect` bbox. Negative answers are always `result=compliant`, `objects=[]`, and
`uncertain=false`. Train prompts may use deterministic semantic variants; validation and test use
one frozen prompt variant.

For one-class Anomalib methods, train only on `train/good`. Select preserved validation and test
rows explicitly from `samples.csv`; do not let a framework synthesize validation from frozen test.
Supervised specialists may also consume train defect rows and their masks from the manifest.

## Specialist result adapter and metric scope

`normalize_specialist_prediction` accepts project fields (`image_score`, `anomaly_map`) and common
Anomalib aliases (`pred_score`, `anomaly_score`, `heatmap`, `pred_mask`). The evaluator applies:

- a configurable heatmap threshold and 4-connected-component minimum area;
- bbox heatmap-to-tile scaling, original-image translation, and boundary clipping;
- image F1 at a validation-selected threshold after max tile-score aggregation per source image;
- tie-aware source-image AUROC, returning null if either class is absent;
- pixel micro precision/recall/F1/IoU after nearest-neighbor heatmap alignment to the truth tile;
- box precision/recall/F1 and Acc@IoU 0.5 using greedy one-to-one tile-level matching.

Do not tune thresholds on test. Pixel metrics count overlapping tile pixels more than once, so they
are explicitly tile-pixel micro averages. Per-tile machine output includes boxes in both tile and
original-image coordinates.

## Commands

Reproduce the frozen export:

```bash
.venv/bin/python -m src.data.ksdd_tile_cli \
  --source-root data/processed/ksdd_v0 \
  --source-lock configs/data/ksdd_v0.lock.json \
  --output-root data/processed/ksdd_tiled_v1 \
  --record-schema configs/data/ksdd_tile_record.schema.json \
  --sft-record-schema configs/data/ksdd_tile_sft_record.schema.json \
  --sft-answer-schema configs/data/ksdd_sft_answer.schema.json \
  --created-at 2026-08-09T08:30:00+08:00 \
  --tile-sizes 256 384 --reference-root .
```

Evaluate real specialist predictions without loading a model in the evaluator:

```bash
.venv/bin/python -m src.evaluation.specialist_cli \
  --tile-records data/processed/ksdd_tiled_v1/tile_records.jsonl \
  --dataset-root data/processed/ksdd_tiled_v1 \
  --predictions SPECIALIST_TEST_PREDICTIONS.jsonl \
  --output SPECIALIST_TEST_REPORT.json \
  --split test --heatmap-threshold VALIDATION_SELECTED_VALUE \
  --image-threshold VALIDATION_SELECTED_VALUE --box-iou-threshold 0.5
```

Run regression checks:

```bash
.venv/bin/pytest -q tests/data tests/evaluation
.venv/bin/ruff check src/data src/evaluation tests/data tests/evaluation
```

## Pilot-only restriction and risks

The frozen test split has 56 source images but only 9 positive source images. This is too small for
a formal KPI claim, stable positive-class confidence intervals, or reliable model ranking. All
specialist outputs must retain `pilot_only=true`; fixture/mock values must never be reported as
model performance.

Further limitations are the narrow controlled KSDD domain, CC BY-NC-SA non-commercial terms,
tile-overlap weighting in pixel metrics, and threshold sensitivity. A formal gate requires a larger
independent test set with enough positive entities, validation-only threshold selection, and an
unchanged frozen evaluation protocol.
