# Clean evaluation region — labeling workflow

Goal: label EVERY tree in this 250x250 m square so we have an HONEST test set.
The botanical inventory undercounts trees (we measured ~94% of "false positives" are real),
so we cannot trust inventory-based precision/recall. Completeness matters more than size —
label this small area with ZERO gaps.

Region bounds (EPSG:32617): [572783.4, 2844787.2, 573033.4, 2845037.2]
Seeded from 240 baseline predictions + 299 inventory trees.

## Files in this folder
- region_clip.tif            the imagery for this region (GITIGNORED — regenerate with build)
- region.geojson             the region boundary
- region_predictions.geojson candidate boxes from the model (EDIT THESE — ~94% are real)
- region_inventory.geojson   inventory tree points (reference; blue)
- region_inventory_boxes.geojson  inventory boxes (used by --eval only)
- verified_trees.gpkg        EMPTY layer for your final ground truth (draw/keep boxes here)
- ../..​/reports/clean_eval/region_overview.png + region_quadrants.png  printed references

## Open in QGIS
1. Layer > Add Raster Layer…  -> region_clip.tif   (your basemap)
2. Layer > Add Vector Layer…  -> region_predictions.geojson  (red candidate boxes)
3. Layer > Add Vector Layer…  -> region_inventory.geojson    (blue reference points)
4. Layer > Add Vector Layer…  -> verified_trees.gpkg         (your editable GT layer)
   All layers are already EPSG:32617; QGIS should align them on the raster.

## Workflow — build verified_trees so EVERY tree has exactly one box
Toggle editing (pencil) on verified_trees, then for every tree in the imagery:
  - ACCEPT a good prediction: copy the prediction box into verified_trees
    (or draw a box over that crown). status = "kept".
  - REJECT a false box: simply do not add it (leave it only in predictions). We will
    detect deletions automatically by comparing verified_trees to region_predictions.
  - ADD a miss: if a tree has NO prediction box, DRAW a new box. status = "added".
    These are the inventory's misses — the trees we most need captured.
  - status field values (free text is fine): kept / added / adjusted
Save the layer (Ctrl+S) often. There is no wrong pace — re-run the helpers anytime.

## Track progress
    .venv/bin/python src/s11_clean_eval.py --coverage
reports: trees verified, predictions kept vs deleted, brand-NEW trees (no pred + no inventory).

## Get the honest score (once every tree is verified)
    .venv/bin/python src/s11_clean_eval.py --eval
prints precision/recall/F1 of the baseline predictions against your clean GT at IoU 0.4 and
0.3, next to the inventory-based number for the same region — the gap is the eval error.

IMPORTANT: keep these trees OUT of any future training set — this is the held-out honest test.
