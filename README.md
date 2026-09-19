# soflo-tree-models

A playground for image segmentation to detect the trees of South Florida.

## maps

* `./bicy_share.qgs` is a QGIS file that will open if you place the download folder from the [shared Big Cypress folder](https://miami.box.com/s/qd4j3x2pf0v2gn9knjz9r9ffnsyku4wx) at the top level of this repo. You can visualize the cropped images, the approximate survey locations, and the historical survey data.
* `./gables_share.qgs` is a QGIS file that will open if you place the download folder from the [shared Gables Campus folder](https://miami.box.com/s/mq6k0vj8f89h4w91u7pqdocetjpoma2f) at the top level of this repo. You can visualize the drone survey, the tree inventory, and the deep learning models from ArcGIS Pro.

## jupyter

* `./jupyter/crop_orthoimages.ipynb` a code generator that creates a set of shell scripts to crop the original Big Cypress orthophotos to approximate site locations.
* `./jupyter/csv_to_shapefile.ipynb` an example of how to read csv into a geopandas dataframe (from Big Cypress historical survey)

## treedetect

A single config-driven detection pipeline (`s01`–`s20`) run across both sites, plus the
diagnostics that decide whether a score means what it looks like. Everything is
EPSG:32617, metres. The through-line of this branch is **measurement**: the first honest
number here was far worse than the model, and most of the work is establishing which
ruler to trust.

* [./treedetect](treedetect) — the pipeline itself: setup, the point-first labelling
  strategy, the step-by-step run order, and how to reproduce any figure from a committed
  checkpoint without retraining.
* [./treedetect/gables_campus](treedetect/gables_campus) — DeepForest fine-tuned on the
  Coral Gables drone survey. Distance-matched **F1 0.573** at a 5 m radius inside a
  labelled evaluation region (P 0.674 / R 0.498), against a random-scatter floor of 0.178
  recall. Includes the manual FP/FN audit, the false-positive diagnosis, and a 3-way
  architecture benchmark on identical data.
* [./treedetect/bcnp](treedetect/bcnp) — the same playbook carried to Big Cypress wetland
  plots. Three independent detector families all converge to **F1 ≈ 0.11** and barely beat
  random scatter; two dedicated diagnostics rule out georeferencing error and label
  visibility as the cause. The site, not the model, is the wall.
* [./treedetect_methods.md](treedetect_methods.md) — the standalone methods write-up for
  the Gables detection stage: data inputs, label construction, training, and evaluation.

### What the audit changed

The raw IoU-0.4 score was F1 0.393, and it was the wrong ruler on both axes. Hand-tagging
a random sample of the model's false positives in QGIS, tree by tree:

| Of 100 sampled "false positives" | Count |
|---|---|
| Real trees the botanical inventory never recorded | 94 |
| Genuine errors (2 shadows, 1 bush) | 3 |
| Unsure | 3 |

The inventory undercounts trees, so the model was being penalised for finding them.
Splitting the misses tells the mirror-image story: only **5.4%** of false negatives are
true blindness — 94.6% have a box on or near the tree that simply falls below the IoU cut.
The model finds the trees; it boxes them loosely. Distance-matching at a stated radius,
scored inside a labelled region and reported against a random floor, is the correction.

### 3-way architecture benchmark (Gables, 5 m distance match, same region and protocol)

| Model | Pretraining | Precision | Recall | F1 |
|---|---|---|---|---|
| YOLO26s (`origin/ahsan`) | COCO | 0.647 | **0.674** | **0.660** |
| DeepForest fine-tuned (`s19`) | NEON canopy | **0.674** | 0.498 | 0.573 |
| From-scratch heatmap CNN (`s20`) | none | 0.369 | 0.488 | 0.420 |

`s20` is a hand-written ~2M-parameter encoder–decoder CNN — no pretrained weights, no
torchvision backbone, no detection library — predicting a tree-centre heatmap trained with
CenterNet focal loss on point supervision. With data, split, and protocol held identical,
the spread reads as a pretraining ablation worth **~0.15–0.24 F1**. Its recall nearly
matches DeepForest (0.488 vs 0.498); the entire gap is precision.

### Why Big Cypress does not work

| | Gables | BCNP |
|---|---|---|
| Land cover | sparse, urban, palm-heavy | dense wetland canopy |
| Stem spacing | trees metres apart | ~1.6 m, wall-to-wall |
| Background chips | plentiful (controls precision) | none exist inside the plots |
| Distance metric | discriminating at 5 m | saturated by ~1.5 m |
| Edge over random | +0.32 recall | +0.02 recall |

Both comfortable excuses were tested and rejected: recall is flat at ~17% regardless of
tree size (`s17`, so not a visibility ceiling), and the recall surface peaks sharply and
exactly at the current census placement across a full offset/rotation/flip sweep (`s18`,
so not a georeferencing bug). A separate species probe (`s13`) found pine, cypress, and
palm are not separable from overhead RGB crops — neither a linear probe nor a fine-tuned
ResNet18 beats the 72.1% predict-all-pine base rate. Height data (CHM / LiDAR) is the
principled next lever.

## Resources

* [The root shared Box folder](https://miami.box.com/s/qtijqu7e5g36wircuzts951mk8oqwyj3)
  * [Running Notes](https://miami.box.com/s/86s4e45lwxvta9lh5ow5mlze72ngh5v9)
  * [The shared Gables Campus folder](https://miami.box.com/s/mq6k0vj8f89h4w91u7pqdocetjpoma2f)
  * [The shared Big Cypress Folder](https://miami.box.com/s/qd4j3x2pf0v2gn9knjz9r9ffnsyku4wx)
  * [The GDSC metadata spreadsheet](https://miami.box.com/s/cpe136whxprafac9ssvkig74ju4o2x7m)
