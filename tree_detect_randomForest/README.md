# treedetect — tree detection from drone orthomosaics

Detection-only pipeline for marking trees in high-resolution drone imagery,
piloted on the UM Coral Gables campus and intended to transfer to Big Cypress.
Stage 1 of a two-stage plan (detection now; species classification later).

Everything is in **EPSG:32617 (UTM 17N)**, units in metres.

## Results & campaign write-ups

The headline results and findings live in per-site summaries (read these first):

- **[`gables_campus/README.md`](gables_campus/README.md)** — Gables DeepForest results.
  Distance-matched **F1 0.573** (5 m, clean-eval region; P 0.674 / R 0.498), the manual
  FP/FN audit (94/100 false positives were real unrecorded trees; recall miss is loose
  boxes, not blindness), resolution tuning, and a head-to-head vs Ahsan's YOLO26s (0.66).
- **[`bcnp/README.md`](bcnp/README.md)** — Big Cypress wetland results (`s13`–`s18`).
  Three detector families all converge to **F1 ≈ 0.11** and barely beat random scatter;
  dedicated diagnostics rule out a georeferencing bug (`s18`) and a label-visibility
  ceiling (`s17`). The site, not the model, is the wall.

## The key data finding (why labels are built the way they are)

We have three layers for campus:

| layer | what it is | trustworthy? |
|---|---|---|
| `um_gables_trees.geojson` | 10,659-point botanical inventory; 6,988 confirmed trees (`point_is_tree == "Yes"`), most with species | **yes — ground truth for location** |
| `um_gables_tree_segmentation.geojson` | prior model's crown polygons | no |
| `um_gables_tree_detection.geojson` | prior model's boxes | reference only |

Joining the confirmed points to the polygons shows the prior model **found only
~42% of real trees** (3,092 of 6,988 land inside any polygon) and **merged
adjacent crowns** (640 polygons contain >1 tree). That is the "overlapping
canopies / shadows" failure mode, quantified.

So detection labels are built **point-first**: every confirmed tree gets one box.
A clean 1-tree polygon donates its real crown extent; everywhere else we use a
data-driven fixed box (median clean-crown size, ~7 m). Each box is tagged with
`src` — `polygon`, `merged_fixed`, or `nomatch_fixed` — so provenance is auditable.
See `reports/label_qa.png`.

## Pipeline

| step | script | needs | output |
|---|---|---|---|
| 01 | `s01_build_labels.py` | the 3 geojsons | `outputs/detection_boxes.geojson`, `reports/label_qa.png` |
| 02 | `s02_define_aois.py` | step 01 | AOI + per-AOI box geojsons, `reports/aoi_split.png` |
| 03 | `s03_prepare_tiles.py` | **the ortho `.tif`** | clipped rasters + tiled DeepForest annotations |
| 04 | `s04_train.py` | step 03 (GPU helps) | fine-tuned checkpoint |
| 05 | `s05_evaluate.py` | step 04 | precision / recall on the held-out test AOI |
| 06 | `s06_predict.py` | step 04 | `outputs/predicted_trees.geojson` for QGIS |

Steps 01–06 above are the Gables detection core. Later steps extend it:

| steps | what | write-up |
|---|---|---|
| `s07`–`s08` | species labels + crop-classifier probe (Gables) | — |
| `s09`–`s12` | recall experiments, manual FP/FN audit, clean-eval region, FP diagnosis | `gables_campus/README.md` |
| `s19` | Gables distance-matched headline F1 | `gables_campus/README.md` |
| `s13` | BCNP 3-class species probe | `bcnp/README.md` |
| `s14`–`s16` | BCNP detection: DeepForest boxes, heatmap peak-finder, YOLO26s | `bcnp/README.md` |
| `s17`–`s18` | BCNP diagnostics: size-visibility re-score, georeferencing check | `bcnp/README.md` |

Steps 01–02 run anywhere. Steps 03–06 need the orthomosaic and `deepforest`/`torch`.
The BCNP steps (`s13`–`s18`) need the Big Cypress plots + census under `~/Downloads/bcnp/`.

## Reproduce / Setup

The input data is **not** in the repo — each user supplies their own copy and
points the pipeline at it via `SOFLO_DATA_ROOT`. Outputs (tiles, checkpoints,
figures) are written inside the repo.

```bash
# 1. clone + enter the pipeline
git clone https://github.com/tibbben/soflo-tree-models.git
cd soflo-tree-models/treedetect

# 2. install dependencies (Python 3.11 recommended; deepforest 2.x + torch)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. point at your local data copy
cp .env.example .env
#   then edit .env so SOFLO_DATA_ROOT=/your/path/to/soflo_data
```

**Expected data directory** (`$SOFLO_DATA_ROOT`, flat layout — filenames must
match `config.yaml` `inputs:`):

```
soflo_data/
├── umgables_2025_drone_survey_5cm.tif      # orthomosaic — RGB, 5 cm/px, EPSG:32617
├── um_gables_trees.geojson                 # botanical inventory points (ground truth)
├── um_gables_tree_segmentation.geojson     # prior-model crown polygons (reference)
├── um_gables_tree_detection.geojson        # prior-model boxes (reference only)
└── GDSC_metadata(Metadata).csv             # GDSC tile metadata catalog
```

`SOFLO_DATA_ROOT` resolves in this order: **env var** → **`data_root:` in
config.yaml** → a clear error telling you to set it. The orthomosaic + the three
geojsons (~3.7 GB + ~39 MB) are large and intentionally excluded from git.

**Run order** (s03–s06 need only the orthomosaic; s01–s02 also need the geojsons):

```bash
python src/s01_build_labels.py    # point-first labels  -> outputs/detection_boxes.geojson
python src/s02_define_aois.py     # spatial train/test split + aoi_split.png
python src/s03_prepare_tiles.py   # clip ortho to AOIs, tile + annotate
python src/s04_train.py           # fine-tune DeepForest (MPS/GPU helps); TD_EPOCHS overrides
python src/s05_evaluate.py        # precision/recall on the test AOI (best checkpoint)
python src/s06_predict.py         # predict over test AOI -> outputs/predicted_trees.geojson
```

To regenerate figures from an existing checkpoint without retraining, run
s03/s05/s06 only (s05/s06 load the best checkpoint via `common.best_checkpoint`).

## Configuration — `config.yaml`

- **AOIs**: by default the train AOI is the bottom-left corner and test is the
  top-right (the dense, clean regions). For exact extents, draw rectangles in
  QGIS, read the bounds, and set `aois.<name>.bounds: [minx, miny, maxx, maxy]`.
- **Tile size** is set on the ground (`tiles.patch_size_m`, default 40 m) and
  converted to pixels from the ortho's GSD at runtime — robust to whatever
  resolution your survey is. DeepForest's released weights expect ~10 cm/px; if
  your ortho is much finer, this is the knob that keeps crowns a sensible size.

## Notes / decisions to revisit

- The ortho is assumed 3-band RGB; step 03 takes the first 3 bands. The inventory
  has an `ndvi_average` field, hinting a NIR band may exist — worth checking, as
  NIR helps separate canopy from shadow.
- Species labels have spelling variants (e.g. *Swietenia mahagoni* vs
  *Switenia mahagoni*) — clean these before the classification stage.
- Validation in step 04 reuses the train set for a first pass; point it at the
  test tiles once you trust the loop.
