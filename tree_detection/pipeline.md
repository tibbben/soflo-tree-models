\# Tree Detection Pipeline



A YOLOv8 computer-vision pipeline that detects and maps individual trees across a

high-resolution drone survey. It tiles a large orthomosaic into chips, fine-tunes a

detector on labeled tree points, runs full-survey inference, and scores the result

against ground-truth tree locations.



Single class only (`Tree`); no species classification in the output. Every detection

is exported as a point (the centroid of its predicted box) in the survey's projected

CRS (EPSG:32617).



All scripts are run from the project root and use relative paths.



\---



\## Data layout



```

download/

&#x20; <survey>.tif              # drone orthomosaic (GeoTIFF, 8-bit RGB)

&#x20; <trees>.geojson           # ground-truth tree points

yolo\_dataset/

&#x20; images/{train,val}/       # generated chips (PNG)

&#x20; labels/{train,val}/       # generated YOLO box labels (TXT)

&#x20; dataset.yaml              # YOLO dataset config (paths + class names)

runs/detect/runs/train1-\*/  # training outputs (weights/best.pt, results.csv)

output/                     # full-survey detections (GeoJSON + shapefile)

scripts/                    # the scripts below

```



\---



\## Scripts and run order



The core flow is: \*\*chip → split → train → detect → benchmark.\*\*



\### 1. `chip\_data.py` — tile the survey and write labels

Tiles a GeoTIFF into fixed-size chips and writes a matching YOLO box label per chip,

building a square box around each ground-truth tree point at a fixed crown radius.

Background (tree-free) chips are sampled at a fixed ratio.



```

python scripts/chip\_data.py full        # whole survey -> images/train

```



Key settings inside the file: `CHIP\_SIZE`, `OVERLAP`, the crown radius, and the

survey path. \*\*These define the scale the model is trained at.\*\*



\### 2. `split\_dataset.py` — train/val split

Randomly moves 20% of the chips into the validation set (fixed seed for

reproducibility). Run once after chipping.



```

python scripts/split\_dataset.py

```



\### 3. `train\_yolo.py` — fine-tune the detector

Fine-tunes a YOLOv8 model on the chips. Writes to an auto-incremented run folder

under `runs/detect/runs/train1-N/`. `best.pt` is the epoch with the highest

validation mAP50; `last.pt` is the final epoch. Validation runs automatically at the

end of every epoch and drives early stopping.



```

python scripts/train\_yolo.py

```



\### 4. `detect\_full.py` — full-survey inference (the deliverable)

Tiles the entire survey window-by-window (memory-safe), runs the trained model on

each tile, converts pixel boxes to world coordinates, removes tile-overlap duplicates

with cross-tile NMS, and exports one point per tree to GeoJSON + shapefile.



```

python scripts/detect\_full.py <weights> \[conf] \[nms\_iou]

```



\*\*Its tiling settings (`CHIP\_SIZE`, `OVERLAP`, `IMGSZ`) must match the values used in

`chip\_data.py` when those weights were trained.\*\* A mismatch makes the model see

tiles at the wrong scale and produces wrong detections.



\### 5. `benchmark.py` — score against ground truth

Point-matches detections to the ground-truth points (one-to-one, highest-confidence

first, nearest within the match radius) and reports precision / recall / F1. Works on

points or polygons (polygons reduced to centroids). An optional confidence floor lets

you sweep operating points from a single low-confidence detection file.



```

python scripts/benchmark.py <detections.geojson> \[match\_dist\_m] \[min\_conf]

```



\---



\## Metrics: two different numbers



\- \*\*Validation mAP50\*\* (from training / `scan\_results.py`) is box-IoU based on

&#x20; held-out chips. Use it to steer training (early stopping, picking `best.pt`). It is

&#x20; \*\*not comparable across runs\*\* that used different survey resolution or box sizes.

\- \*\*Full-survey F1\*\* (from `benchmark.py` / `benchmark\_all.py`) is point-distance

&#x20; based over the whole survey at a chosen match radius (5 m and 3 m). This is the

&#x20; metric that decides which model is best, because it matches the point deliverable.



Confidence threshold does not change the model — it only picks the precision/recall

operating point. Compare models each at its own peak-F1 confidence (sweep with the

`min\_conf` argument), not at a fixed threshold.



\---



\## Operational notes



\- \*\*Close any GIS application before running `detect\_full.py` / `benchmark\_all.py`.\*\*

&#x20; An open viewer can lock the output GeoJSON/shapefile, so the write silently fails

&#x20; and you benchmark stale results.

\- \*\*Inference tiling must mirror training tiling.\*\* When benchmarking older weights,

&#x20; set `detect\_full.py` to the survey/chip/overlap those weights were trained with.

\- \*\*Run folders auto-increment.\*\* A new training run writes to the next free

&#x20; `train1-N`; always use the path the training step prints rather than assuming the

&#x20; number.

\- \*\*Match the channel count.\*\* The survey is read as RGB (`\[:3]`); a 4th band, if

&#x20; present, is dropped.

\- Ground truth may be incomplete, which understates precision (some "false positives"

&#x20; are real, unlabeled trees) and caps recall by labeling rather than by the model.

