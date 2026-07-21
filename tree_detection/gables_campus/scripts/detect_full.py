"""
detect_full.py — config-driven detection across the FULL survey. This produces the
actual deliverable: a tree point layer covering the entire site.

Companion to detect.py, which runs the same model over the evaluation-region crop for
benchmarking. The two scripts differ in exactly two things: which raster they read
(data.survey vs data.survey_gtregion) and which file they write. All tiling geometry
comes from the shared config, so neither can drift from the chipper or from each other.

Use detect.py for any number you intend to compare against a recorded result.
Use this script for the map you hand over.

Run from the project root ON A GPU. The full survey is several times as many tiles as
the evaluation region, so submit this as a batch job rather than an interactive session
(an interactive session dies with the SSH connection).

    python scripts/detect_full.py <config.yaml> <weights> [conf]

Output: ./output/tree_detections_full.geojson (+ .shp)
"""

import sys
import os
import numpy as np
import torch
from torchvision.ops import nms
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Point
from ultralytics import YOLO

sys.path.insert(0, "./scripts")
from config import load, summary

if len(sys.argv) < 3:
    sys.exit("Usage: python scripts/detect_full.py <config.yaml> <weights> [conf]")
cfg = load(sys.argv[1])
summary(cfg, "detect_full")
weights_path = sys.argv[2]

# For the deliverable, pick the confidence deliberately — it is the precision/recall
# operating point, not a model setting. The champion peaks at F1 with 0.15; 0.10 finds
# noticeably more trees at similar F1 if coverage matters more than a clean layer.
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

d = cfg["data"]
OUT_SIZE = d["out_size"]
GROUND_M = d["ground_m"]
OVERLAP_M = d["overlap_m"]
IMGSZ = cfg["detect"]["imgsz"]
NMS_IOU = cfg["detect"]["nms_iou"]

out_dir = "./output"
os.makedirs(out_dir, exist_ok=True)

# distinct filename from the benchmarking run, so a full-survey pass can never
# overwrite detections that a recorded score was computed from
out_geojson = f"{out_dir}/tree_detections_full.geojson"
out_shp = f"{out_dir}/tree_detections_full.shp"


def offsets(total, size, step):
    # tile start positions, with a final offset flush to the raster edge
    offs = list(range(0, total - size, step))
    if not offs or offs[-1] != total - size:
        offs.append(total - size)
    return offs


model = YOLO(weights_path)
world_boxes = []
world_confs = []

# THE ONLY GEOMETRY DIFFERENCE FROM detect.py: the full survey, not the crop
with rasterio.open(d["survey"]) as src:
    raster_crs = src.crs
    pixel_size = src.res[0]
    src_chip_px = int(round(GROUND_M / pixel_size))
    src_overlap_px = int(round(OVERLAP_M / pixel_size))
    step = src_chip_px - src_overlap_px

    row_offs = offsets(src.height, src_chip_px, step)
    col_offs = offsets(src.width, src_chip_px, step)
    total_tiles = len(row_offs) * len(col_offs)
    print(f"FULL survey {src.width}x{src.height}px @ {pixel_size:.4f}m/px "
          f"-> {total_tiles} tiles ({src_chip_px}px -> {OUT_SIZE}px) @ conf {conf}. "
          f"Detecting...")

    done = 0
    skipped = 0
    for row_off in row_offs:
        for col_off in col_offs:
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total_tiles} tiles | {len(world_boxes)} detections "
                      f"| {skipped} empty tiles skipped")

            window = Window(col_off, row_off, src_chip_px, src_chip_px)
            tile = src.read([1, 2, 3], window=window,
                            out_shape=(3, OUT_SIZE, OUT_SIZE),
                            resampling=Resampling.bilinear)

            # skip near-empty tiles (nodata / outside the survey footprint)
            if np.mean(tile) < 5:
                skipped += 1
                continue

            # the detection framework expects BGR channel order
            tile_bgr = np.ascontiguousarray(np.transpose(tile, (1, 2, 0))[:, :, ::-1])
            results = model.predict(tile_bgr, imgsz=IMGSZ, conf=conf, verbose=False)
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes_px = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            win_t = src.window_transform(window)
            left = win_t.c
            top = win_t.f

            for (x1, y1, x2, y2), c in zip(boxes_px, confs):
                # output-chip px -> world coordinates
                wx1 = left + x1 / OUT_SIZE * GROUND_M
                wx2 = left + x2 / OUT_SIZE * GROUND_M
                wy1 = top - y1 / OUT_SIZE * GROUND_M
                wy2 = top - y2 / OUT_SIZE * GROUND_M
                world_boxes.append([min(wx1, wx2), min(wy1, wy2),
                                    max(wx1, wx2), max(wy1, wy2)])
                world_confs.append(float(c))

print(f"Tiles processed: {total_tiles - skipped} of {total_tiles} "
      f"({skipped} empty, skipped)")
print(f"Raw detections (with tile-overlap duplicates): {len(world_boxes)}")
if len(world_boxes) == 0:
    sys.exit("No detections — check the weights path and confidence threshold.")

# cross-tile NMS: the same tree seen in two overlapping tiles becomes one detection
keep = nms(torch.tensor(world_boxes, dtype=torch.float32),
           torch.tensor(world_confs, dtype=torch.float32), NMS_IOU).numpy()
print(f"After cross-tile NMS: {len(keep)} unique trees")

# one POINT per tree, at the box centroid
points = [Point((world_boxes[i][0] + world_boxes[i][2]) / 2,
                (world_boxes[i][1] + world_boxes[i][3]) / 2) for i in keep]
out_confs = [world_confs[i] for i in keep]

gdf = gpd.GeoDataFrame({"confidence": out_confs}, geometry=points, crs=raster_crs)
gdf.to_file(out_geojson, driver="GeoJSON")
gdf.to_file(out_shp)
print(f"Wrote {len(gdf)} trees to {out_geojson} (+ .shp)")