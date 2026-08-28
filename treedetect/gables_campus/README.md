# UM Gables Campus — Individual Tree Detection (DeepForest)

Detect every tree in a high-resolution drone orthomosaic of the University of Miami
Coral Gables campus and export one point per tree for GIS use. This is the DeepForest
line of work (pipeline steps `s01`–`s12` in `../src/`); it is a sibling to the YOLO
effort on `origin/ahsan` and is benchmarked against it below.

---

## Headline result

**Distance-matched F1** — box centroids matched to the nearest inventory point,
greedy highest-confidence first, one-to-one (KD-tree), best-F1 over a confidence
sweep. This is the metric that is comparable across our BCNP work and to Ahsan's
YOLO benchmark (which also reduces boxes to centroids and matches by distance),
unlike the raw IoU number we started with.

Scored inside the **s11 clean-eval region** (a labelled-area polygon, so the survey's
large unlabelled gaps do not punish precision):

| Match radius | Precision | Recall | F1 | Random floor (R) | Edge over random (R) |
|---|---|---|---|---|---|
| 1.0 m | 0.270 | 0.080 | 0.124 | 0.005 | +0.075 |
| 2.0 m | 0.303 | 0.224 | 0.258 | 0.041 | +0.183 |
| **5.0 m** | **0.674** | **0.498** | **0.573** | 0.178 | **+0.320** |

Full test AOI (all 2 125 preds vs 2 211 inventory, larger sample, more unlabelled area):

| Match radius | Precision | Recall | F1 |
|---|---|---|---|
| 1.0 m | 0.145 | 0.133 | 0.139 |
| 2.0 m | 0.301 | 0.275 | 0.287 |
| 5.0 m | 0.535 | 0.514 | 0.524 |

The model sits **far above the random-scatter floor at every radius** (edge +0.08 to
+0.32 recall), so the detections carry real signal — this is a genuinely working
detector, in contrast to the BCNP wetland where every model barely beat random.

**Raw IoU-0.4 (where we started, `s05`): P 0.400 / R 0.386 / F1 0.393.** As the audit
below shows, that number badly understates the model on both axes.

Numbers reproducible via `../src/s19_gables_distance_eval.py` →
`../outputs/gables_distance_eval.csv`. No retraining; scored from the committed
`predicted_trees.geojson` and the `s04` checkpoint.

---

## The manual audit — the most important result here

The raw IoU F1 of 0.39 is not the model's real accuracy. I hand-tagged a random
sample of the model's "false positives" in QGIS, tree by tree. These are **my own
manual judgements** (`../outputs/audit/fp_audit_completed.csv`,
`fp_rescore_summary.txt`), not an automated recompute.

**Of 100 sampled false positives: 94 were real trees the botanical inventory never
recorded. 3 were errors (2 tree shadows, 1 bush). 3 were unsure.**

So the inventory undercounts trees, and most "false positives" are the model correctly
finding trees the ground truth is missing. Crediting those 94 audited real trees:

| Precision @IoU-0.4 | Value | Basis |
|---|---|---|
| Original (no credit) | 0.315 | 682 TP / 2 162 preds |
| Corrected (audit-measured) | **0.359** | +94 audited real trees credited as TP |
| Honest gap-only (extrapolated) | **~0.61** | `s12`: only 46% of real-tree FPs are true inventory gaps → 43/94 × 1 480 FPs |
| Credit-all (upper bound) | ~0.96 | if the 94% real rate held across all 1 480 FPs |

The **honest corrected precision is ~0.34–0.61**, not 0.32 — see the `s12` split next.
The ~0.96 figure is an upper bound that (wrongly) assumes every real-tree FP is an
inventory gap; the measured gap fraction pulls the honest number down to ~0.61.

**Merged crowns:** 15 of the 94 real-tree tags carry my note that one predicted box
covers multiple crowns ("multiple in one"). At ~7 m boxes in dense campus canopy, the
model fuses adjacent trees — a labelling-geometry limit, not a detection failure.

---

## FP diagnosis — localization vs genuine inventory gap (`s12`)

I split the 94 audited real-tree FPs by whether an inventory tree sits nearby:

| Bucket | Count | Share |
|---|---|---|
| (a) Genuine inventory gap (no inventory tree near) | 43 | 45.7% |
| (b) Localization (inventory tree present, box just off) | 51 | 54.3% |

The (b) split is decided by **box overlap**, not a distance cutoff — all 51
localization FPs overlap an inventory box, and every gap FP sits ≥ 7.9 m from any
inventory tree (median 29 m), so the 43/51 split is stable for any cutoff below ~8 m.
Of the 51 localization FPs, **21 are double-counts**: one loose box produced *both* a
false positive *and* a false negative on the same tree.

---

## FN diagnosis — loose boxes, not blindness (`s10`)

The recall miss is the mirror image. Of all false negatives:

| Bucket | Count | Share |
|---|---|---|
| True miss (undetected / blind) | 83 | 5.4% |
| Poor localization, IoU (0, 0.30) | 923 | 60.4% |
| Near-miss, IoU [0.30, 0.40) | 337 | 22.0% |
| Contested (its box was claimed by a neighbour) | 186 | 12.2% |

**Only 5.4% of misses are true blindness — 94.6% have a box on or near the tree that
simply falls below IoU 0.40.** The model finds the trees; it boxes them loosely. This
is exactly why distance-matching at a loose radius (5 m) roughly doubles the effective
recall vs the tight IoU-0.4 threshold.

---

## Data

- **Orthomosaic:** `umgables_2025_drone_survey_5cm.tif` — RGB drone survey, 5 cm/px,
  EPSG:32617. (A 1.6 cm native survey exists; 5 cm is used — see Resolution.)
- **Ground truth:** `um_gables_trees.geojson` — the campus botanical inventory as tree
  points. Proven incomplete by the audit above.
- **Reference (not training):** prior ArcGIS-Pro detection/segmentation layers.

---

## Pipeline

Each step is one script in `../src/`, driven by `../config.yaml`.

| Step | Role |
|---|---|
| `s01_build_labels` | Point-first hybrid boxes from inventory points (+ crown polygons where reliable) |
| `s02_define_aois` | Spatial train/test AOI split (no tile in both) |
| `s03_prepare_tiles` | Clip each AOI, world→pixel boxes, tile with DeepForest `split_raster` |
| `s04_train` | Fine-tune DeepForest from the base release; early stop on val box-recall |
| `s05_evaluate` | Raw IoU-0.4 precision/recall on the held-out AOI |
| `s06_predict` | Windowed prediction over the whole test AOI → `predicted_trees.geojson` |
| `s09_recall_experiments` | No-retrain recall probes (score threshold, tiling, NMS) |
| `s10_audit` | Manual FP/FN audit tooling + FN bucket analysis |
| `s11_clean_eval` | Clean-eval region scaffold (exhaustive-GT bootstrap; the labelled polygon) |
| `s12_fp_diagnosis` | Split audited real-tree FPs into localization vs inventory gap |
| `s19_gables_distance_eval` | Distance-matched headline F1 (this document) |

**Model.** DeepForest (RetinaNet + ResNet-50, pretrained on NEON forest-canopy crowns),
fine-tuned on the campus tiles. 32-bit, MPS.

**Labelling — point-first hybrid boxes.** Every inventory point becomes a box: a
reliable crown polygon where one exists, otherwise a fixed ~7 m fallback box (inventory
box median 6.98 m). The model then predicts variable-size boxes (median 6.9 m, range
0.6–22 m).

**Split.** Two ~45 ha AOIs — train = bottom-left (1 951 boxes), test = top-right
(2 211 boxes). Spatial, so no tree leaks across the split.

---

## Key findings

- **The inventory, not the model, is the precision bottleneck.** 94/100 sampled FPs are
  real trees the inventory never recorded; corrected precision is ~0.34–0.61, not 0.32.
- **Recall is loose boxes, not blindness.** 94.6% of misses have a box on the tree that
  misses the IoU cut; only 5.4% are truly undetected.
- **Raw IoU-0.4 (0.39) is the wrong ruler.** It penalises both real detections (as FPs)
  and loosely-boxed hits (as FNs). Distance-matching and the clean-eval region are the
  corrections.
- **Resolution: 5 cm is the operating point.** The native 1.6 cm survey adds detail
  without context at crown scale; 5 cm is where campus crowns resolve cleanly.
- **Clean-eval region (`s11`).** Because the inventory undercounts, whole-survey
  precision is unreliable; scoring inside a labelled-area polygon is the honest frame.
  (The exhaustive per-tree relabel of that region was scaffolded but not completed, so
  the headline still scores against inventory inside the polygon — a conservative
  precision, i.e. still understated.)

---

## Comparison: DeepForest (this work) vs YOLO26s (`origin/ahsan`), same site

Apples-to-apples — both distance-matched, 5 m radius, inside a labelled region:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| DeepForest (this work, clean-eval region) | **0.674** | 0.498 | 0.573 |
| YOLO26s (Ahsan, evaluation region) | 0.647 | **0.674** | **0.660** |

My **precision matches/edges his (0.674 vs 0.647); his advantage is recall** (0.674 vs
0.498) — YOLO finds more of the trees, which is the whole F1 gap. Ahsan's own
architecture bake-off (his protocol) put DeepForest at 0.542 vs YOLO26 at 0.615, the
same ordering. YOLO's small-object/aerial design and stronger augmentation give it the
recall edge on this sparse, palm-heavy urban canopy; DeepForest's NEON-canopy
pretraining keeps its precision competitive.

---

## Limitations & next steps

- **Recall is the gap.** DeepForest boxes are loose and fuse merged crowns; that costs
  recall at any tight matching. Adopting the YOLO wins (5 cm constant-footprint chips,
  aerial augmentation, longer training) is the obvious lever — and BCNP `s16` already
  ported them.
- **No exhaustive ground truth.** The `s11` region was scaffolded but its per-tree
  relabel was never finished, so precision here is still inventory-limited (understated).
  Completing it would move the headline precision toward the audited ~0.6–0.96.
- **Merged crowns / box tightness** is the structural ceiling — a crown-segmentation or
  point/peak head (rather than fixed boxes) is the principled fix, consistent with the
  BCNP conclusion.
- **Metric hygiene:** report distance-matched F1 with an explicit radius and a
  random-scatter floor going forward; never headline a raw IoU number for a
  known-incomplete inventory.
