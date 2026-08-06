# Big Cypress — Progress Summary

Exploratory individual-tree detection on the Big Cypress plot clips, using models
developed for and pretrained on other sites. Output is one point per detected crown,
EPSG:32617, for GIS use.

**This site has no usable ground truth yet**, so nothing here is benchmarked. Every
conclusion below comes from visual inspection in QGIS. Numbers are reported for scale,
not as performance.

---

## Status of ground truth

A plot census exists (685 tagged trees across the 54 plots, converted to a GeoJSON point
layer), but **the census points do not align with the imagery** — there is a positional
offset between the derived tree coordinates and the crowns visible in the orthomosaics.
Until that is resolved, precision/recall/F1 cannot be computed for this site.

Consequence: confidence thresholds cannot be chosen by maximising a metric, as they were
on campus. They were chosen by eye, and should be treated as provisional.

---

## Data

51 per-plot orthomosaic clips in `./download/ortho_clipped/`.

| Property | Value |
|---|---|
| Ground sample distance | 1.69 cm |
| Clip size | ~9466 × 9466 px = 160 m square (2.56 ha) |
| Bands | 4 (uint8) — band 4 is **alpha, constant 255**, not NIR |
| CRS | EPSG:32617 |

As on campus, no near-infrared signal is available, so vegetation-index approaches remain
closed.

**Three plots produced no detections from either model**: `plot_11_2`, `plot_12_3`,
`plot_8_3`. Because both models failed identically, this is a data property of those
clips rather than a model or pipeline issue. 48 of 51 plots yielded output.

---

## Method

Two models were run, each fed imagery at the resolution it was trained at. The clips are
downsampled on the fly from the 1.69 cm source — no intermediate files are created.

| | Model | Tile geometry | Effective GSD |
|---|---|---|---|
| YOLO | Campus champion (YOLO26, fine-tuned) | 640 px over 32 m | 5 cm |
| DeepForest | `weecology/deepforest-tree` release weights | 400 px over 40 m | 10 cm |

The campus champion was trained on 5 cm imagery in 640 px / 32 m tiles; the DeepForest
release model was trained on NEON imagery at ~10 cm in 400 × 400 px patches. Matching each
model to its own training geometry is more defensible than forcing both through identical
tiling, because neither is being fine-tuned here and so nothing would correct a scale
mismatch.

This does mean the two models did **not** see identical inputs, unlike the campus
architecture comparisons where resolution was held constant.

**The campus DeepForest fine-tunes were deliberately not used.** Fine-tuning specialised
them toward a managed campus, which is the wrong direction for this site. The unmodified
release weights are the better-matched starting point.

Downsampling uses area-averaging, not bilinear interpolation. Bilinear samples sparse
points when shrinking an image and aliases, manufacturing spurious edge texture across
continuous canopy. An initial YOLO run using bilinear produced 55,738 detections; the same
run with averaging produced 69,528 — the cleaner imagery yielded *more* detections, not
fewer, indicating the earlier resampling was degrading crown structure rather than
inflating detections.

---

## Results

Detection counts at a 0.05 confidence floor, across 48 plots (2.56 ha each):

| Model | Total | Per plot |
|---|---|---|
| YOLO champion | 69,528 | ~1,448 |
| DeepForest release | 40,197 | ~837 |

Counts surviving each confidence threshold:

| Threshold | YOLO | DeepForest |
|---|---|---|
| 0.05 | 69,528 | 40,197 |
| 0.10 | 44,776 | 37,699 |
| 0.15 | 28,656 | 35,191 |
| 0.20 | 19,178 | 31,951 |
| 0.25 | 13,458 | 27,846 |
| 0.30 | 9,775 | 21,848 |
| 0.40 | 5,411 | 10,774 |
| 0.50 | 2,885 | 3,957 |

The *shape* of these distributions is more informative than the totals. From 0.05 to 0.20,
YOLO loses 72% of its detections while DeepForest loses only 20%. YOLO is producing a long
tail of low-confidence guesses; DeepForest is confident about most of what it finds. That
is the signature of a well-matched domain versus a poorly-matched one.

Note that the two confidence scales are **not comparable** — a score of 0.25 means
different things to the two models, and each needs its own threshold.

---

## Visual findings

**DeepForest performs better on this site.** This reverses the campus ranking, where
YOLO26 beat DeepForest (F1 0.660 vs 0.542). The reversal is expected: DeepForest is
pretrained on natural forest canopy, which is far closer to Big Cypress wetland forest
than to a manicured campus.

**The clearest evidence is bare-crowned trees.** Pale, leafless crowns visible in the
imagery are detected by DeepForest and **missed entirely by the YOLO champion**. The
likely explanation is that the campus training set contains no such trees: campus species
are evergreen (palms, live oaks), whereas this site is dominated by bald cypress, which is
deciduous and appears bare and pale from above outside the growing season. The champion
has no learned representation for a leafless crown.

This is a structural limitation, not a threshold or tuning problem. No confidence setting
recovers a class of object the model was never trained to recognise.

**The YOLO champion over-detects in closed canopy.** It fires repeatedly across
continuous canopy, sometimes placing two or three points on a single crown. The mechanism
is the training domain: the model learned isolated crowns separated by lawn and pavement,
and closed canopy presents no such gaps. This was confirmed to be domain shift rather than
a resampling artifact, since correcting the resampling increased detections rather than
reducing them.

**Provisional thresholds** (chosen visually, not validated):

- YOLO champion: 0.20–0.25
- DeepForest: lower thresholds looked better; ~0.15 and below

---

## Outputs

Per-plot and merged point layers for each model:

```
output/
├── yolo/<plot>.geojson              # one file per plot
├── yolo_all_plots.geojson           # merged — load this in QGIS
├── deepforest/<plot>.geojson
└── deepforest_all_plots.geojson     # merged — load this in QGIS
```

Every point carries a `confidence` attribute and a `plot` label.

Saved QGIS projects, one per model and threshold:

```
yolo_champion_0.15conf.qgz     yolo_champion_0.20conf.qgz     yolo_champion_0.25conf.qgz
deepforest_pretrained_0.05conf.qgz  deepforest_pretrained_0.10conf.qgz
deepforest_pretrained_0.15conf.qgz
```

**Changing the threshold does not require re-running detection.** Detection is run once at
a 0.05 floor and the confidence is stored per point, so any higher threshold is a filter.
In QGIS: right-click the layer → Filter (or Properties → Source → Query Builder), and
enter e.g.

```
"confidence" > 0.25
```

Adjust the number freely to compare operating points live. The saved projects above are
simply this filter applied at different values.

---

## Tooling

Scripts in `scripts/`, run from the project root:

| Script | Role |
|---|---|
| `config.py` | Shared config loader; imported, not run directly |
| `detect_yolo.py` | YOLO detection across all plot clips |
| `detect_deepforest.py` | DeepForest release-model detection across all plot clips |

Configs in `configs/`: `bigcypress_plots.yaml` (YOLO geometry) and
`bigcypress_plots_deepforest.yaml` (DeepForest geometry).

`detect_yolo.py` and `detect_deepforest.py` require different environments — the
DeepForest one is pinned to `deepforest==1.5.2` and `albumentations<2.0`.

---

## Compute

Detection for this site was run entirely on a personal laptop: RTX 4060 Laptop GPU
(8 GB VRAM), 32 GB RAM, Intel Core i7-13700HX. The plot clips total tens of gigabytes,
so transferring them to a cluster would have cost more time than the inference itself.

Inference is far lighter than training — the full 51-plot YOLO pass takes well under an
hour on this hardware, and DeepForest is comparable. By contrast, the campus model these
runs use was trained on the University of Miami Pegasus cluster (NVIDIA H100, IBM LSF
scheduler); no training was performed for this site.

---

## Future work

1. **Fix the census alignment.** This is the blocker for everything quantitative. Until
   the plot census registers correctly against the imagery, no model can be scored here
   and all threshold choices remain subjective.

2. **Label a subset of this site.** Even a few fully-annotated plots would allow real
   benchmarking and, more importantly, fine-tuning a model on this domain rather than
   transferring one.

3. **NMS-IoU tuning.** Both detectors currently use an IoU of 0.5, inherited from the
   campus pipeline and never swept there either. Closed canopy is exactly the case where a
   lower value should help, since it merges the stacked duplicates seen on single crowns.
   Published work on urban canopy detection has favoured much lower values (~0.2). This is
   untested and worth trying, though without ground truth it can only be judged visually.

4. **Crown box size.** The campus work established 5 m as optimal there, bracketed by 3 m
   and 7 m. Note this is *not* a setting that can be changed on this site's pipeline — box
   size is fixed in the trained weights, so testing it at Big Cypress means either
   retraining on campus at a different size and transferring, or waiting for labels here.
   Given that this site's trees differ substantially in size from campus ornamentals, it is
   plausible the campus-optimal box size is wrong for this domain.

5. **Segmentation rather than detection.** Merged canopy is the dominant failure mode at
   this site, and it is a label-geometry problem that bounding boxes cannot solve.
   Instance segmentation (e.g. detectree2) is the better-matched approach for dense
   forest, and DeepForest's own strength lies in crown segmentation rather than point
   detection.

