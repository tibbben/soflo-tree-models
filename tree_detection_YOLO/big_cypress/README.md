# Big Cypress — Exploratory Tree Detection

Individual-tree detection on the Big Cypress plot clips, using models developed for and
pretrained on other sites. One point per detected crown, EPSG:32617.

**Status: exploratory.** No usable ground truth exists for this site yet, so nothing here
is benchmarked and every conclusion is from visual inspection.

## Why there are no metrics

A plot census exists — 685 tagged trees across the 54 plots — but the census points do not
align with the imagery. Until that offset is resolved, precision, recall and F1 cannot be
computed here, and confidence thresholds have to be chosen by eye rather than by
maximising a score.

## What was run

Two models, each fed imagery at the resolution it was trained at, downsampled on the fly
from the 1.69 cm source clips:

| Model | Tile geometry | Effective GSD | Detections @ 0.05 |
|---|---|---|---|
| Campus champion (YOLO26, fine-tuned) | 640 px over 32 m | 5 cm | 69,528 |
| DeepForest release weights | 400 px over 40 m | 10 cm | 40,197 |

**DeepForest looks better here**, reversing the campus ranking. The clearest evidence is
bare, pale crowns that DeepForest detects and the campus champion misses entirely — the
campus training set is all evergreen species, so the model has no representation for a
leafless crown. See `progress_summary.md`.

## Contents

| File | What it is |
|---|---|
| [pipeline.md](pipeline.md) | How to run both detectors: environments, configs, commands |
| [progress_summary.md](progress_summary.md) | Data, method, results, findings, and future work |
| `configs/` | One YAML per model — the two differ in tile geometry, deliberately |
| `scripts/` | Two detectors and a shared config loader |
| `*.qgz` | QGIS projects, one per model and confidence threshold |

## Changing the confidence threshold

Detection is run once at a 0.05 floor with the confidence stored on every point, so any
higher threshold is a filter — no re-running required. In QGIS, right-click the layer →
Filter, and enter:

```
"confidence" > 0.25
```

The saved `.qgz` projects are this filter applied at different values.

## Where this was run

Both detectors were run on a laptop — NVIDIA RTX 4060 (8 GB VRAM), 32 GB RAM, Intel Core
i7-13700HX. Detection is inference only, so 8 GB is comfortable, and the plot clips are
large enough (~370 MB each, ~18 GB total) that moving them to a cluster costs more time
than the compute saves. The campus weights used here were trained separately on the
University of Miami Pegasus HPC cluster (NVIDIA H100, IBM LSF batch scheduler).

## Input data

Not tracked in git. The `.qgz` projects load the detection points (committed) over the
plot clips, which are not committed — place the download folder from the
[shared Big Cypress folder](https://miami.box.com/s/qd4j3x2pf0v2gn9knjz9r9ffnsyku4wx) at
`./download/` so the clips sit at `./download/ortho_clipped/plot_*.tif`. Without them the
projects open with the tree points rendering and the imagery layers unavailable.
