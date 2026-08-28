# UM Gables Tree Detection — Methods

**Repository:** `soflo-tree-models` · **Branch:** `rayan` · **Pipeline:** `treedetect` (s01–s06)
**Study area:** University of Miami, Coral Gables campus
**Stage:** Detection (stage 1 of 2; species classification is stage 2, post-detection)
**Status:** Full training run complete (50-epoch cap; early-stopped at epoch 13 on a val-recall plateau; ~28 min on Apple M3 Pro / MPS). Best checkpoint = epoch 5. Clear improvement over the earlier 3-epoch smoke test, but still recall-limited — see §6.

---

## 1. Purpose

Replace the manual tree-marking process with an automated two-stage pipeline: **detect** individual trees first, then **classify** species (palm, cypress, pine). This document covers the detection stage only. The same approach is intended to extend from UM Gables to the BCNP / Racoon Point imagery.

## 2. Data inputs

All design decisions were made by inspecting the actual files first, not from assumptions.

| Input | Contents | Role |
|---|---|---|
| `um_gables_trees.geojson` | 10,659 botanical inventory points | **Authoritative ground truth** |
| `um_gables_tree_detection.geojson` | 5,872 machine-generated boxes (prior) | Reference only |
| `um_gables_tree_segmentation.geojson` | 5,883 polygons (prior) | Reference only — **unreliable** |
| Orthomosaic (`.tif`) | 3-band RGB, 5 cm/px, **no NIR band** | Imagery |
| GDSC metadata catalog (CSV) | Tile provenance / naming | Context |

**Key finding driving the labeling strategy:** the prior segmentation output missed ~58% of confirmed inventory trees and merged adjacent crowns in ~640 polygons. This is the quantified version of the known failure modes (overlapping canopies, shadows). Conclusion: inventory **points** are trustworthy; prior **polygons** are not. Labels are therefore built point-first.

## 3. Label construction (point-first, hybrid boxes)

For every inventory point, generate a bounding box:

1. **Where a clean single-tree segmentation polygon exists** for that point, use the polygon's extent as the box (best fidelity).
2. **Otherwise**, fall back to a data-derived fixed box (~6.9 m, derived from the clean-polygon size distribution, not guessed).

Each label carries a **provenance tag** recording which path produced it, so QA and later filtering can distinguish polygon-derived from fixed-size boxes.

**Output:** 6,988 labeled trees across the inventory-derived label set.

## 4. Spatial train / test split

The campus AOI is split **spatially** (geographically separated train and test clips) rather than by random point sampling. This prevents spatial leakage — the model is evaluated on ground it never saw in training. Crowns near the split boundary are handled so the same tree cannot appear in both sets.

- **Train AOI clip:** 1,951 label boxes
- **Test AOI clip:** 2,211 label boxes

(The two clips are subsets of the 6,988 total; they do not cover the entire campus.)

**Validation source.** Training uses the train AOI; validation/early-stopping uses the
held-out **test AOI** (the README's "point it at the test tiles once you trust the loop").
Because the site has only two AOIs, the same test set is used both to select the checkpoint
(best val recall) and to report final metrics, so the reported test numbers are mildly
**optimistic** (selection bias). A future run should carve a third validation block to keep
the test set fully blind.

## 5. Pipeline steps

| Step | Does |
|---|---|
| `s01` | Build point-first hybrid labels with provenance tags |
| `s02` | Spatial train/test split; QA figures (`label_qa.png`, `aoi_split.png`) |
| `s03` | Clip orthomosaic to train/test AOIs; overlay label boxes for visual QA |
| `s04` | Train detector (DeepForest 2.x) |
| `s05` | Predict on the test AOI |
| `s06` | Evaluate predictions vs inventory ground truth; metrics + figures |

Every stage emits a PNG figure for direct visual inspection.

## 6. Model & training (full run)

- **Detector:** DeepForest 2.x (RetinaNet, NEON-pretrained), fine-tuned on the UM Gables labels.
- **Hardware / device:** Apple M3 Pro (18 GB unified memory), **MPS** backend, 32-bit precision (no AMP), `PYTORCH_ENABLE_MPS_FALLBACK=1`, `batch_size=2`.
- **Schedule:** 50-epoch cap with **early stopping** on validation `box_recall` (patience 8). Checkpointing every epoch retains **best-by-val-recall** and **last**; final eval uses the best checkpoint.
- **Outcome:** early-stopped at **epoch 13**; **best epoch = 5**. Total wall-clock **27.9 min**, ~**118–129 s/epoch**. `train_loss` falls while `val_loss` climbs after ~epoch 2 → overfitting on the 200-tile train set; early stopping halts at the recall plateau. See `reports/train_curve.png` and `outputs/train_metrics.csv`.

**Final test-AOI metrics (best checkpoint, IoU 0.4):**

| Metric | 3-epoch smoke test | **Full run (best @ epoch 5)** |
|---|---|---|
| Precision (s05, score≥0.30) | 0.449 | **0.400** |
| Recall (s05, score≥0.30) | 0.275 | **0.386** |
| F1 (s05, score≥0.30) | 0.341 | **0.393** |
| Best val recall (in-training, raw) | — | **0.464** (epoch 5) |
| Predicted boxes (test AOI, s06) | 1,388 | **2,162** |
| Inventory trees (test AOI) | 2,211 | 2,211 |

Recall improved from 0.28 → 0.39 (+~40% relative) and predicted count moved from 1,388 toward the 2,211 inventory total. **Two threshold regimes** are reported because they differ: the in-training `box_recall` (0.464) is computed on the raw model output, while the s05/s06 figures apply the deployment confidence cut `score_thresh=0.30`, which trims low-confidence boxes (lower recall, higher precision). The detector is meaningfully better than the smoke test but still misses ~3 in 5 trees at the deployment threshold — the small single-AOI training set and the 5 cm/px vs ~10 cm/px pretraining mismatch are the leading suspects.

## 7. QA / visualization outputs

- `label_qa.png` — labels overlaid on imagery to confirm boxes land on canopies.
- `aoi_split.png` — train vs test geographic separation.
- `s03` clip figure — train and test AOI clips with point-first label boxes (raster downsampled for display; all boxes drawn).
- `s06` evaluation figures — predictions vs ground truth.

## 8. Environment / reproducibility

- Python; DeepForest 2.x (PyTorch backend); ReportLab; openpyxl (catalog work).
- Geospatial I/O: GeoTIFF orthomosaics, GeoJSON labels.
- Version control: `soflo-tree-models`, branch `rayan`, committed in discrete steps.

## 9. Known limitations & open questions (for PI meeting)

- **RGB-only, no NIR** — limits vegetation separability; a Canopy Height Model (CHM) from photogrammetry is the most promising lever for the shadow / overlapping-canopy failure modes.
- **Undertrained baseline** — 3 epochs only; longer training + prediction tiling/threshold tuning needed before drawing model conclusions.
- **Box-size fallback** — fixed ~6.9 m boxes may under/over-size for very large or very small crowns; affects IoU-based recall.
- **Model choice** — DeepForest is NEON-forest-pretrained; an urban dense-canopy site may favor a site-trained YOLO or a Mask R-CNN (Detectree2) for crown polygons. Worth a controlled comparison.

## 10. Next steps

1. ✅ Longer training done (50-epoch cap, early-stopped at epoch 13). Still open: tune prediction patch size / `score_thresh`, and — most importantly — **enlarge / augment the training set**, since the detector overfits a single 200-tile AOI within ~2 epochs.
2. Controlled model comparison (DeepForest fine-tuned vs YOLOv8/11 vs Detectree2) on the same split.
3. Integrate CHM as an added channel / filter.
4. SAM-based crown segmentation prompted by detection boxes (stage after detection, not a standalone finder).
5. Marker-controlled watershed seeded by inventory points to bootstrap polygon supervision.
6. Extend pipeline to BCNP / Racoon Point.
