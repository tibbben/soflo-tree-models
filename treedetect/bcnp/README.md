# Big Cypress (BCNP) — Wetland Tree Detection & Species Probe

Transfer the Gables detection work to Big Cypress National Preserve: detect individual
trees in dense wetland canopy and probe whether the three dominant species are separable.
Ground truth is the historical plot census (stems as points, with DBH / species / alive
flag). Steps `s13`–`s18` in `../src/`.

**One-line result:** BCNP is a genuinely hard site. Three independent detector families
(DeepForest boxes, a heatmap peak-finder, and YOLO26s) all converge to the same weak
number — barely above a random-scatter floor at honest tight radii — and dedicated
diagnostics rule out the two comfortable excuses (it is **not** a georeferencing bug and
**not** a label-visibility artifact). The bottleneck is the canopy itself.

---

## Headline result

Detection, held-out test plots (`plot_9_3` + `plot_12_1`), **distance-matched** (box/peak
centroids → nearest census point, greedy top-confidence, one-to-one, KD-tree) with a
**random-scatter floor** (same count, uniform in plot). Distance eval — not IoU —
because the ground truth is points and the crowns are tiny.

| Detector (step) | radius | Recall | Precision | F1 | Random F1 | Edge (recall) |
|---|---|---|---|---|---|---|
| DeepForest boxes (`s14`, IoU-0.4) | — | 0.039 | 0.132 | 0.060 | — | — |
| Heatmap peak-finder (`s15`) | 1.0 m | 0.160 | 0.088 | 0.113 | 0.100 | +1.9 pts |
| YOLO26s (`s16`) | 1.0 m | 0.165 | 0.083 | **0.110** | 0.096 | +2.1 pts |
| YOLO26s (`s16`) | 1.5 m | 0.188 | 0.094 | 0.126 | 0.126 | **+0.0 pts** |

**All three model families land at F1 ≈ 0.11–0.13 at a 1 m radius, and every one of them
is within a few points of random scatter.** At 1.5 m the metric is so saturated by tree
density (~1 500–2 600 stems/ha, ~1.6 m spacing) that random points score the same as the
model. This is the opposite of Gables, where the same DeepForest architecture sits +0.32
recall above random.

For contrast, the Gables DeepForest model on its own site (distance-matched, 5 m,
labelled region) reaches **F1 0.573**. Same code, same author — the difference is the
site.

---

## Species probe (`s13`)

Before detection, a decoupled test: given a perfectly-centred crop of a census tree, can a
CNN tell pine / cypress / palm apart? 20 254 alive crops from 8 clean plots (pine 72.1 %,
cypress 14.5 %, palm 13.5 %), spatial split by plot.

| Model | Test accuracy | vs base rate (predict-all-pine 72.1 %) |
|---|---|---|
| Frozen-ResNet18 linear probe | 0.423 | −29.8 pts |
| Fine-tuned ResNet18 (class-weighted) | 0.569 | −15.2 pts |

**Neither beats the 72.1 % base rate.** Cypress collapses into pine (recall 0.15; 75 % of
cypress predicted as pine). From 1.5 m RGB crops the species are not reliably separable —
a ceiling on any species-aware detector here. Palm (radial fronds) is the only visually
distinct class, and even it recalls only 0.20.

---

## Detection experiments

### `s14` — DeepForest from base (box regression, first run)

Census points → DBH-allometry boxes (not the Gables 7 m box), each ~1.6 cm/px plot
**resampled to 10 cm/px** (to match the base release's training domain) and masked to its
100 m plot polygon so unlabelled margin cannot become a phantom false negative. Trained
from the base release (not the Gables checkpoint — different domain).

- IoU-0.4 recall **3.9 %**, IoU-0.3 recall **7.1 %** (precision 0.13 / 0.24).
- FN buckets: **99 % box-tightness, only 0.9 % true blindness** — the model fires roughly
  in the right places but boxes too loosely to clear IoU 0.4.
- Gotcha logged: DeepForest 2.1 does not propagate `config.score_thresh` to the RetinaNet
  head at predict time; this low-confidence model (peaks ~0.25) returned zero detections
  at the default 0.30 until the head threshold was set directly.

### `s15` — Point peak-finding (Ventura-style heatmap)

A light resnet18 U-Net regresses a Gaussian tree-confidence heatmap; trees are read off as
local peaks (NMS by min-distance), matched by distance. 320 px tiles at native ~1.6 cm/px,
50 % overlap.

- Weighted MSE **collapsed** to a near-zero map (foreground/background imbalance);
  switching to **CenterNet penalty-reduced focal loss** made it learn (val loss 10.8 → 4.24).
- Distance eval: **F1 0.113 @1 m** (R 0.160 / P 0.088), 0.143 @2 m. Peak confidences are
  low (~0.02), so the eval auto-selects the best-F1 threshold from a sweep.
- Edge over random: **+1.9 pts recall @1 m, ~0 @2 m.** Over-predicts ~1.8× census.

### `s16` — YOLO26s (Ahsan's Gables playbook, adapted)

Ported the portable Gables wins (constant ground-footprint chips, fixed-size boxes, strong
aerial augmentation, long training) and swept them for BCNP, scored with our honest
tight-radius protocol.

- **Best config: 5 cm/px working resolution, 2.0 m fixed box radius.** The resolution
  finding (finer hurts) matches Gables; the box size is 2 m not Ahsan's 5 m — 5 m would
  swallow neighbours at BCNP density. Both picked on the val plot.
- Distance eval: **F1 0.110 @1 m** (R 0.165 / P 0.083), 0.126 @1.5 m — statistically
  tied with the heatmap peak-finder.
- Edge over random: **+2.1 pts @1 m, +0.0 @1.5 m.**
- **Zero background chips exist** (`bg=0` for every config): the plots are so dense that no
  tree-free tiles occur inside the polygon, so the negatives lever that fixed Gables
  over-prediction does not apply, and the model still over-predicts ~2×.

The architecture swap that won Gables (YOLO26s beat DeepForest 0.615 vs 0.542 there) does
**not** rescue BCNP.

---

## What we ruled out

### `s17` — Is the ceiling label visibility? (NO — re-score, no retraining)

Re-scored the `s16` predictions against DBH-filtered census subsets. If the misses were
small/invisible stems, recall would climb sharply for larger trees.

| DBH subset | target | Recall @1 m | Edge over random |
|---|---|---|---|
| all alive | 4 114 | 16.5 % | +2.1 pts |
| DBH > p50 | 2 040 | 17.0 % | +2.2 pts |
| DBH > p75 | 1 019 | 18.0 % | +2.5 pts |
| DBH > p90 | 133 | 17.3 % | +3.5 pts |

**Recall is flat at ~17 % regardless of tree size**, and barely above random even for the
biggest crowns. (TOTHT cuts appear to rise, but their edge over random falls to zero — that
is density, not signal; and TOTHT is only ~17 % populated.) The model over-predicts 2×–62×
the target across subsets. **The ceiling is not visibility — the model does not
preferentially find even canopy-sized trees.**

### `s18` — Is the census→imagery georeferencing offset / rotated / flipped? (NO)

Three independent checks:

- **Palm test (human-verifiable):** high-zoom crops with only *Sabal palmetto* points over
  imagery. The points sit on and among the radial-frond palm crowns with no consistent
  directional displacement.
- **Offset/rotation sweep:** using the `s16` predictions, recall @1 m peaks **sharply and
  exactly at the current placement** — best translation +1 m (+0.24 pts), best rotation 0°,
  and every flip / swapped-XY / 90-180-270° rotation is equal-or-worse. The recall(dx,dy)
  surface is a clean central peak (not flat), so the predictions carry georef signal and
  that signal confirms the current placement.
- **Internal structure:** all 8 plot polygons are exactly axis-aligned 100 × 100 m squares
  (edge deviation 0.0000°), SW corner = origin; census fills X[0–100] Y[0–100] m,
  near-isotropic — no internal rotation.

**Georeferencing is correct.** The shared failure is the task, not the coordinates.

---

## Key findings

- **Dense wetland canopy defeats 2D RGB detection at honest resolution.** DeepForest,
  heatmap peaks, and YOLO all converge to F1 ≈ 0.11 @1 m and barely beat random.
- **The distance metric is density-saturated.** At ~1.6 m spacing, a 1.5–2 m match radius
  lets random scatter score ~19 % recall, so absolute recall numbers are near-meaningless
  without the random floor — always report edge-over-random here.
- **It is not a georef bug (`s18`) and not label visibility (`s17`).** Both comfortable
  explanations were tested and rejected.
- **Species are not separable from 1.5 cm RGB (`s13`).** Cypress and pine look alike from
  above; only palms are distinct.
- **5 cm is the working resolution** (finer adds detail without context at crown scale) —
  the one Gables finding that transferred cleanly.

## Why Gables works and BCNP does not

| | Gables | BCNP |
|---|---|---|
| Land cover | sparse, urban, palm-heavy | dense wetland canopy |
| Stem spacing | trees metres apart | ~1.6 m, wall-to-wall |
| Background chips | plentiful (controls precision) | none (`bg=0`) |
| Distance metric | discriminating at 5 m | saturated by ~1.5 m |
| Edge over random | +0.32 recall | +0.02 recall |

The Gables playbook is sound; BCNP is simply outside what a 2D box/point detector on RGB
can resolve.

## Confirmed dead ends

- DeepForest box regression at native or 10 cm (`s14`) — box-tightness, not fixable by
  threshold or NMS.
- Heatmap peak-finding with weighted MSE — collapses to zero (use focal loss).
- YOLO negatives / background chips — none exist inside dense plots.
- Blaming georeferencing or small-tree invisibility — both ruled out.
- Species-aware detection from RGB — species not separable (`s13`).

## Next steps

- **Height data (CHM / LiDAR)** is the principled lever — it disambiguates merged canopies
  directly, which is exactly BCNP's regime. This is the same conclusion Ahsan reached for
  Gables' merged-canopy cases, and it matters more here. Pending confirmation of LiDAR
  coverage for the site.
- **Crown segmentation** rather than detection, if a per-tree point layer is not strictly
  required.
- If staying with RGB detection, **stop reporting bare distance recall** — headline
  edge-over-random at a stated tight radius, so a density artifact is never mistaken for
  skill.
