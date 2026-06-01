# treedetect — tree detection from drone orthomosaics

Detection-only pipeline for marking trees in high-resolution drone imagery,
piloted on the UM Coral Gables campus and intended to transfer to Big Cypress.
Stage 1 of a two-stage plan (detection now; species classification later).

Everything is in **EPSG:32617 (UTM 17N)**, units in metres.

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

Steps 01–02 run anywhere. Steps 03–06 need the orthomosaic and `deepforest`/`torch`.

## Run

```bash
pip install -r requirements.txt
# put the ortho at data/umgables_2025_drone_survey.tif (see config.yaml)
python src/s01_build_labels.py
python src/s02_define_aois.py
python src/s03_prepare_tiles.py
python src/s04_train.py
python src/s05_evaluate.py
python src/s06_predict.py
```

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
