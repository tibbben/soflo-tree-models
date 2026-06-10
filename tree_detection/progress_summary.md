# GDSC Tree Detection — Run Comparison

Single-class ("Tree") YOLOv8 detection on 5cm UM Coral Gables drone survey imagery. Dataset: full campus tiled into 640px chips, random 80/20 train/val split, 10,659 georeferenced ground-truth points.

## Top 3 runs

| Rank | Run | mAP50 | mAP50-95 | Precision | Recall | Model | imgsz | Batch | Box sizing |
|------|-----|-------|----------|-----------|--------|-------|-------|-------|------------|
| 1 | Run 13 | **0.511** | 0.281 | 0.589 | 0.404 | yolov8s | 1280 | 2 | fixed 5m |
| 2 | Run 14 | 0.410 | 0.208 | 0.457 | 0.413 | yolov8s | 1280 | 2 | per-tree (`canopy_area_sqm`) |
| 3 | Run 6 | 0.376 | 0.129 | 0.414 | 0.379 | yolov8s | 640 | 8 | fixed 5m |

Common settings: patience=30, overlap 100px, single class "Tree".

## Key findings

- **Best lever was resolution:** imgsz 640 -> 1280 raised mAP50 from 0.376 to 0.511 (trees are small objects, so more pixels per tree helps most).
- **Per-tree canopy boxes underperformed:** only ~11% of trees have a real `canopy_area_sqm` value, so variable box sizes added noise rather than signal — a fixed, consistent box trained better. Bottleneck is label completeness, not box sizing.