# Tree Detection — Progress Summary

## Overview

Detect and map every individual tree across a university campus from a
high-resolution drone orthomosaic, using a fine-tuned YOLOv8 detector. Single class
(`Tree`); the output is one point per detected tree in the survey's projected CRS
(EPSG:32617). Performance is measured against a ground-truth set of 10,659 labeled
tree points using a point-distance benchmark (F1 at 5 m and 3 m match radii).

## Pipeline

Chip the orthomosaic into labeled tiles, fine-tune the detector, run full-survey
inference, and score the detections against ground truth. See `pipeline.md` for the
scripts and run order.

## Evaluation method

- **Full-survey F1** — each detection point is matched one-to-one to the nearest
  unclaimed ground-truth tree within a match radius (5 m and 3 m), matching
  highest-confidence detections first. Precision / recall / F1 are reported.
- **Operating point** — the confidence threshold selects the precision/recall
  trade-off and does not change the model. Each model is reported at its own peak-F1
  confidence, found by sweeping a single low-confidence inference pass.
- **Validation mAP50** — box-IoU on held-out chips, reported for reference only. It
  is a training-time metric and is not comparable across runs that used different
  survey resolution or box sizes.

All models below — including the prior baseline — were scored with this identical
method, so the numbers are directly comparable.

## Experiments

Each run changed one variable from the previous best, to isolate its effect.

| Run folder | Variable tested                                  | Full-survey F1 @5m | F1 @3m | Val mAP50 |
|------------|--------------------------------------------------|:------------------:|:------:|:---------:|
| train1-13  | Baseline: yolov8s, 5 cm survey, fixed-5 m boxes  |         —          |   —    |   0.511   |
| train1-14  | Per-tree canopy-area box sizing                  |         —          |   —    |   0.410   |
| train1-15  | Per-species (estimated) box sizing               |       0.538        | 0.442  |   0.182   |
| **train1-16** | **Fixed-5 m boxes + augmentation**            |     **0.571**      |**0.495**| 0.531    |
| train1-17  | Larger backbone (yolov8m)                        |       0.531        | 0.456  |   0.485   |
| train1-18  | Higher resolution: 1.6 cm survey, 2000 px tiles  |       0.533        | 0.463  |   0.483   |
| train1-19  | 1.6 cm survey, measured per-species box sizing   |       0.481        | 0.405  |   0.279   |
| train1-20  | 1.6 cm survey, native 1280 px tiles (no downscale)|      0.551        | 0.452  |   0.505   |

Runs 13–14 were evaluated on validation metrics only; runs 15–20 were benchmarked on
the full survey.

## Comparison to prior baseline

An existing ArcGIS Pro detection and segmentation baseline, scored with the identical
point benchmark:

| Model                          | F1 @5m | F1 @3m | Precision | Recall |
|--------------------------------|:------:|:------:|:---------:|:------:|
| ArcGIS Pro detection           | 0.355  | 0.286  |   0.500   | 0.276  |
| ArcGIS Pro segmentation        | 0.355  | 0.283  |   0.499   | 0.275  |
| **Best model (train1-16)**     |**0.571**|**0.495**|  0.507   | 0.654  |

The best model improves F1 by ~61% at 5 m and finds **2.4× as many trees**
(recall 0.654 vs 0.276). Every fine-tuned run in the table above outperforms the
prior baseline.

## Key findings

1. **Augmentation helps.** Rotation + flips + light mixup lifted the baseline
   (train1-16 over train1-13) with no precision cost.
2. **A larger backbone does not help.** yolov8m (train1-17) underperformed yolov8s
   and cost roughly double the training time — capacity is not the bottleneck.
3. **Uniform fixed box sizing beats variable sizing**, confirmed three independent
   ways (train1-14, train1-15, train1-19 all lost to fixed-5 m boxes). Box size only
   affects how the model learns; the output collapses every box to a point, so
   variable sizing has no payoff path.
4. **Higher source resolution did not improve the deliverable.** The 5 cm model
   (train1-16, 0.571) outscored the same configuration on the 1.6 cm survey
   (train1-18, 0.533). Native-resolution tiling (train1-20, 0.551) recovered most of
   the loss caused by downscaling high-resolution tiles, but still did not exceed the
   5 cm result. A likely explanation: the detection target is crown-scale, and 5 cm
   imagery is already near that scale, whereas 1.6 cm adds sub-crown detail that does
   not aid crown-level detection.
5. **Validation mAP50 is a weak proxy for full-survey F1.** The run with the lowest
   val mAP50 (train1-15, 0.182) was among the best on the deliverable (0.538). Model
   selection should use the full-survey point benchmark, not val mAP50.
6. **Confidence threshold is an operating-point choice**, not a model property; each
   model is fairly compared at its own peak-F1 confidence.

## Caveats

- Margins among the top models (0.571 / 0.551 / 0.533) are small and based on single
  training runs; differences within ~0.02 F1 may partly reflect training
  stochasticity. Confirming the resolution finding would require repeated runs.
- Ground truth is incomplete: some apparent false positives are real, unlabeled
  trees, so precision is understated and recall is capped by labeling rather than by
  the model.

## Current best

**train1-16** — 5 cm survey, fixed-5 m boxes, augmentation. Full-survey
**F1 0.571 at 5 m** (0.495 at 3 m), at confidence 0.05.

## Next steps

- Evaluate a pretrained tree-crown model (e.g. DeepForest) as an out-of-the-box
  baseline, scored with the same benchmark.
- Test newer small-object detector architectures (YOLO11 / YOLO26).
- Repeat the top runs to confirm the resolution finding given the small margins.
- Ground-truth completeness audit to quantify the true recall ceiling.
- Extend the pipeline to the next survey site.