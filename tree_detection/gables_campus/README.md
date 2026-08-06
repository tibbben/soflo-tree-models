# Gables Campus — Individual Tree Detection

Detect and map every tree on the Coral Gables campus from a drone orthomosaic, exporting
one point per tree in EPSG:32617 for GIS use.

## Result

| | Precision | Recall | F1 |
|---|---|---|---|
| Champion — YOLO26, 5 cm, fixed 5 m boxes | 0.647 | 0.674 | **0.660** |
| Prior ArcGIS Pro baseline, same region | 0.738 | 0.275 | 0.401 |

Roughly a 65% improvement in F1 and 2.45× as many trees found. The baseline is
high-precision and low-recall — conservative, missing about three quarters of the trees —
while the objective here is to find every tree, so recall is the metric that matters.

Scored on an evaluation region rather than the full survey, because the ground truth does
not cover the whole site. See `progress_summary.md`.

## Contents

| File | What it is |
|---|---|
| [pipeline.md](pipeline.md) | How to reproduce the champion: setup, configuration, and every command |
| [progress_summary.md](progress_summary.md) | What was tried, what won, what was ruled out, and why |
| `configs/` | One YAML per experiment — a run is defined entirely by its config |
| `scripts/` | The nine-file pipeline: tile, split, train, detect, benchmark |
| `best_results.qgz` | QGIS project visualising the champion's full-survey detections |

## Getting started

Read `pipeline.md`. The short version: every stage reads the same YAML config, so a run is
specified in one file and the scripts do not change between experiments.

```bash
python scripts/tile_data.py ./configs/champion_5cm_5m.yaml
python scripts/split_dataset.py
python scripts/train_yolo.py ./configs/champion_5cm_5m.yaml
python scripts/detect_gtregion.py ./configs/champion_5cm_5m.yaml <weights> 0.05
```

## Data

Most input data is not tracked in git. `best_results.qgz` loads the detection points
(committed) over the drone survey, which is not committed — place the download folder from
the [shared Gables Campus folder](https://miami.box.com/s/mq6k0vj8f89h4w91u7pqdocetjpoma2f)
at `./download/` so the survey sits at
`./download/umgables_2025/umgables_2025_drone_survey_5cm.tif`. Without it the project opens
with the tree points rendering and the imagery layer unavailable.

Two small files under `download/` **are** tracked, because no script regenerates them and
every reported figure depends on them: `gt_region.geojson` (the evaluation polygon) and
`um_gables_trees_gtregion.geojson` (the ground truth clipped to it). See §7 of
`pipeline.md`.
