# Tree Detection Pipeline

A YOLOv8 computer-vision pipeline that detects and maps individual trees across a high-resolution drone survey. It tiles a large orthomosaic into chips, fine-tunes a detector on labeled tree points, runs full-survey inference, and scores the result against ground-truth tree locations.

Single class only (`Tree`); no species classification in the output. Every detection is exported as a point (the centroid of its predicted box) in the survey's projected CRS (EPSG:32617).

All scripts are run from the project root and use relative paths.

## Data layout

```text
download/
  survey.tif                 # drone orthomosaic (GeoTIFF, 8-bit RGB)
  trees.geojson              # ground-truth tree points
yolo_dataset/
  images/train, images/val   # generated chips (PNG)
  labels/train, labels/val   # generated YOLO box labels (TXT)
  dataset.yaml               # YOLO dataset config (paths + class names)
runs/detect/runs/            # training outputs (weights/best.pt, results.csv)
output/                      # full-survey detections (GeoJSON + shapefile)
scripts/                     # the scripts below
```

## Scripts and run order

The core flow is: **chip, split, train, detect, benchmark.**

### 1. `chip_data.py` — tile the survey and write labels

Tiles a GeoTIFF into fixed-size chips and writes a matching YOLO box label per chip, building a square box around each ground-truth tree point at a fixed crown radius. Background (tree-free) chips are sampled at a fixed ratio.

```bash
python scripts/chip_data.py full
```

Key settings inside the file: `CHIP_SIZE`, `OVERLAP`, the crown radius, and the survey path. These define the scale the model is trained at.

### 2. `split_dataset.py` — train/val split

Randomly moves 20% of the chips into the validation set (fixed seed for reproducibility). Run once after chipping.

```bash
python scripts/split_dataset.py
```

### 3. `train_yolo.py` — fine-tune the detector

Fine-tunes a YOLOv8 model on the chips. Writes to an auto-incremented run folder under `runs/detect/runs/`. `best.pt` is the epoch with the highest validation mAP50; `last.pt` is the final epoch. Validation runs automatically at the end of every epoch and drives early stopping.

```bash
python scripts/train_yolo.py
```

### 4. `detect_full.py` — full-survey inference (the deliverable)

Tiles the entire survey window-by-window (memory-safe), runs the trained model on each tile, converts pixel boxes to world coordinates, removes tile-overlap duplicates with cross-tile NMS, and exports one point per tree to GeoJSON and shapefile.

```bash
python scripts/detect_full.py WEIGHTS [conf] [nms_iou]
```

Its tiling settings (`CHIP_SIZE`, `OVERLAP`, `IMGSZ`) must match the values used in `chip_data.py` when those weights were trained. A mismatch makes the model see tiles at the wrong scale and produces wrong detections.

### 5. `benchmark.py` — score against ground truth

Point-matches detections to the ground-truth points (one-to-one, highest-confidence first, nearest within the match radius) and reports precision, recall, and F1. Works on points or polygons (polygons reduced to centroids). An optional confidence floor lets you sweep operating points from a single low-confidence detection file.

```bash
python scripts/benchmark.py DETECTIONS [match_dist_m] [min_conf]
```

## Metrics: two different numbers

- **Validation mAP50** (from training) is box-IoU based on held-out chips. Use it to steer training (early stopping, picking `best.pt`). It is **not comparable across runs** that used different survey resolution or box sizes.
- **Full-survey F1** (from `benchmark.py`) is point-distance based over the whole survey at a chosen match radius (5 m and 3 m). This is the metric that decides which model is best, because it matches the point deliverable.

Confidence threshold does not change the model — it only picks the precision/recall operating point. Compare models each at its own peak-F1 confidence (sweep with the `min_conf` argument), not at a fixed threshold.

## Operational notes

- **Close any GIS application before running inference.** An open viewer can lock the output GeoJSON/shapefile, so the write silently fails and you benchmark stale results.
- **Inference tiling must mirror training tiling.** When benchmarking older weights, set `detect_full.py` to the survey, chip size, and overlap those weights were trained with.
- **Run folders auto-increment.** A new training run writes to the next free folder; always use the path the training step prints rather than assuming the number.
- **Match the channel count.** The survey is read as RGB (first 3 bands); a 4th band, if present, is dropped.
- Ground truth may be incomplete, which understates precision (some apparent false positives are real, unlabeled trees) and caps recall by labeling rather than by the model.