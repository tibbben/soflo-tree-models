# Progress Summary

Individual tree detection and mapping from high-resolution drone orthomosaic imagery.
Goal: detect every tree across the survey area and export one point per tree for GIS use.

---

## Headline result

**Champion: YOLO26, 5 cm imagery, fixed 5 m crown boxes, aerial augmentation.**

| | Precision | Recall | F1 |
|---|---|---|---|
| Champion (evaluation region, conf 0.15) | 0.647 | 0.674 | **0.660** |
| Prior ArcGIS Pro baseline (same region) | 0.738 | 0.275 | 0.401 |

A **~65% improvement in F1** over the baseline, and **2.45× as many trees found**
(7185 true positives vs 2936).

The baseline is high-precision and low-recall: conservative, and misses roughly three
quarters of the trees. For a mapping deliverable — where the objective is to find every
tree — recall is the metric that matters, and the champion wins decisively on it.

Full-survey scores are also recorded (champion F1 0.615, baseline 0.355), but see
*Evaluation region* below for why those figures understate every model.

---

## The evaluation-region finding

The most important methodological result of the project.

Large parts of the survey have no ground-truth labels, yet the model correctly detects
real trees there. Scored against the full survey, those correct detections count as false
positives, understating precision — and F1 — for every model tested, the baseline
included. Inspecting the supposed false positives in GIS confirmed they are trees.

The fix: clip the survey and the ground truth to a polygon covering only the labelled
area, and evaluate inside it. Effect on the champion:

| | Precision | Recall | F1 |
|---|---|---|---|
| Full survey | 0.559 | 0.685 | 0.615 |
| Evaluation region | 0.647 | 0.674 | **0.660** |

Recall barely moves; precision carries the entire gain. The correction is conservative —
unlabelled trees *inside* the polygon still count against precision — so true precision is
likely higher still. All benchmarking is performed on the evaluation region.

---

## Architecture

All at 5 cm, fixed 5 m boxes, identical data and augmentation (full-survey F1, from
before the evaluation-region correction):

| Architecture | F1 |
|---|---|
| YOLO26 | **0.615** |
| YOLOv8s | 0.571 |
| YOLO11s | 0.551 |
| DeepForest, best variant | 0.542 |

YOLO26's advantage comes from its small-object/aerial design, not from being newer —
YOLO11s scored *below* YOLOv8s. A larger backbone within the older family (YOLOv8m,
0.531) did not help either.

---

## Resolution

Constant-input design (every chip emitted at a fixed pixel size covering a fixed ground
footprint; coarser sources upsampled into it, so image detail is the only variable).
Evaluation-region F1:

| Resolution | Peak F1 | Precision | Recall |
|---|---|---|---|
| 5 cm | **0.660** | 0.647 | 0.674 |
| 7.5 cm | 0.649 | 0.625 | 0.674 |
| 10 cm | 0.645 | 0.609 | 0.685 |
| 20 cm | 0.624 | 0.608 | 0.642 |

Monotonic, with an accelerating decline past 10 cm. From 5 to 10 cm the loss is entirely
in precision (recall flat) — detail helps reject non-crowns, not find them. At 20 cm
recall drops too, as crowns stop resolving well enough to locate.

Finer than 5 cm does not help: 1.6 cm was tested twice and lost both times (0.533, 0.551).
5 cm is already at crown scale. **Resolution is closed.**

An earlier resolution test scaled the model input with source resolution, which upscaled
every chip to the same size and normalised detail away — producing a flat, uninformative
result. The constant-input design above corrects that; any future resolution comparison
must use it.

---

## Crown box size

All YOLO26 at 5 cm, evaluation-region F1:

| Box radius | Peak F1 | Peak conf | Precision | Recall |
|---|---|---|---|---|
| 3 m | 0.584 | 0.075 | 0.589 | 0.579 |
| **5 m** | **0.660** | 0.15 | 0.647 | 0.674 |
| 7 m | 0.631 | 0.10 | 0.571 | 0.704 |

A clean inverted U with 5 m at the peak, bracketed on both sides, and the two failures are
mirror images:

- **7 m fails on precision** (0.571). Boxes large enough to enclose a crown also enclose
  shrubs, shadows and roof structures.
- **3 m fails on recall** (0.579). Tighter boxes produce weaker, lower-confidence
  detections and miss larger crowns; the whole confidence curve shifts down.

5 m sits where the box exceeds every measured crown radius on site (1.7–3.35 m) without
being large enough to swallow non-tree structures.

**Box size is closed.** Intermediate sizes (2 m, 4 m) were not pursued — any difference
would fall inside a seed-noise band that has not yet been measured.

---

## Variable box sizing (closed)

Sizing boxes per tree or per species failed three separate ways: per-tree canopy boxes
regressed; per-species boxes from literature values scored 0.538; per-species boxes from
measured on-site values scored 0.481 (giant boxes for the large-canopy species dominate
the failure).

The root cause is data coverage. Only ~11% of ground-truth trees carry any usable crown
size. One crown-size field is ~50% "populated" but almost entirely zeros — 0.2% usable.

Attempts to predict crown size from other fields also failed:

- height → crown radius: R² 0.185 globally; adding trunk diameter reaches 0.208
- per species: 0.465 for one palm species (n=39), but 0.004 and 0.015 for the two most
  common species

Managed and pruned trees have their natural size-to-crown relationship severed. Predicting
crown size from tabular fields collapses to per-species averages, which already lost.

The only route to variable boxes covering more than ~11% of trees is image-derived, which
is what the pseudo-box experiment below tested.

---

## DeepForest (closed)

DeepForest (RetinaNet + ResNet50, pretrained on NEON forest canopy crowns), evaluated
as an alternative to YOLO:

| Variant | F1 |
|---|---|
| Fixed-box fine-tune | 0.465 |
| Image-derived pseudo-box fine-tune (5 cm) | **0.542** |
| Image-derived pseudo-box fine-tune (10 cm) | 0.522 |

Pseudo-boxes — the pretrained model's own crown predictions, filtered to those landing on
a real ground-truth point, with a fixed-box fallback — gained +0.077 over fixed boxes.
That is the clearest evidence that image-derived crown geometry beats fixed geometry in
principle. It still lost to YOLO26 (0.660).

An NMS threshold sweep moved F1 by 0.007, indicating the false positives are
wrong-location boxes rather than stacked duplicates.

**Closed for this deliverable.** DeepForest's real strength is crown *segmentation*,
which is a different output than the point layer required here, and remains relevant to
future dense-forest work.

---

## Open problem: merged canopies

The model performs worst on large trees and on congested stands where adjacent canopies
merge into a single mass.

The cause is label geometry, not model capacity. A fixed box centred on a ground-truth
point cannot teach crown extent: a 15 m canopy is labelled with the same small box as a
3 m one, and merged canopies are labelled as several separate boxes inside one visual
blob. The point-based benchmark partly masks this, since boxes are reduced to centroids
before scoring.

Nothing tested so far — resolution, augmentation, architecture, box size — addresses it.
Real fixes are crown-aware:

- Instance segmentation instead of detection (different deliverable)
- A canopy height model from LiDAR as an additional input channel, which would
  disambiguate merged canopies directly. Pending confirmation of whether LiDAR coverage
  exists for the site.

---

## Confirmed dead ends

Do not retry:

- Per-tree canopy-area boxes
- Per-species boxes, from literature or from measured values
- Predicting crown size from height, trunk diameter, or species
- Larger backbone within the older YOLO family
- YOLO11s (scored below YOLOv8s)
- 1.6 cm source resolution (tested twice)
- DeepForest for point-detection F1
- Vegetation-index / near-infrared approaches — the fourth band of the
  highest-resolution survey is an alpha channel, not NIR. No spectral signal is available.
- Longer training without augmentation

---

## Key operating points

The champion at other confidence thresholds, for use cases that weight recall differently:

| Confidence | Precision | Recall | F1 |
|---|---|---|---|
| 0.15 | 0.647 | 0.674 | 0.660 (peak) |
| 0.10 | — | 0.799 | 0.656 |

Confidence selects an operating point; it does not change the model. Comparisons between
models must use each model's own peak-F1 confidence.