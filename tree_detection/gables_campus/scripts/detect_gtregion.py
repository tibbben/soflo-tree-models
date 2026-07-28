"""
detect_gtregion.py — config-driven detection on the evaluation-region crop, for
benchmarking.

Reads the SAME config as the chipper, so the tiling geometry cannot drift between
training and inference.

Companion to detect_full.py, which runs the same model across the whole survey to
produce the deliverable map. The two differ in exactly two things: which raster they
read (data.survey_gtregion vs data.survey) and which file they write.

Use this script for any number you intend to compare against a recorded result.

Run from the project root ON A GPU:
    python scripts/detect_gtregion.py <config.yaml> <weights> [conf]

A low confidence floor lets one detection run cover an entire threshold sweep at
benchmark time.

Output: ./output/tree_detections_gtregion.geojson (+ .shp)
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
    sys.exit("Usage: python scripts/detect_gtregion.py <config.yaml> <weights> [conf]")
cfg = load(sys.argv[1])
summary(cfg, "detect_gtregion")
weights_path = sys.argv[2]
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

d = cfg["data"]
OUT_SIZE = d["out_size"]
GROUND_M = d["ground_m"]
OVERLAP_M = d["overlap_m"]
IMGSZ = cfg["detect"]["imgsz"]
NMS_IOU = cfg["detect"]["nms_iou"]

out_dir = "./output"
os.makedirs(out_dir, exist_ok=True)


def offsets(total, size, step):
    # tile start positions, with a final offset flush to the raster edge
    offs = list(range(0, total - size, step))
    if not offs or offs[-1] != total - size:
        offs.append(total - size)
    return offs


model = YOLO(weights_path)
world_boxes = []
world_confs = []

with rasterio.open(d["survey_gtregion"]) as src:
    raster_crs = src.crs
    pixel_size = src.res[0]
    src_chip_px = int(round(GROUND_M / pixel_size))
    src_overlap_px = int(round(OVERLAP_M / pixel_size))
    step = src_chip_px - src_overlap_px

    row_offs = offsets(src.height, src_chip_px, step)
    col_offs = offsets(src.width, src_chip_px, step)
    total_tiles = len(row_offs) * len(col_offs)
    print(f"Evaluation-region crop {src.width}x{src.height}px @ {pixel_size:.4f}m/px "
          f"-> {total_tiles} tiles ({src_chip_px}px -> {OUT_SIZE}px). Detecting...")

    done = 0
    for row_off in row_offs:
        for col_off in col_offs:
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total_tiles} tiles | {len(world_boxes)} detections so far")

            window = Window(col_off, row_off, src_chip_px, src_chip_px)

            # mirrors chip_data.py exactly — bilinear, because coarser surveys are
            # upsampled into the constant output size rather than downsampled
            tile = src.read([1, 2, 3], window=window,
                            out_shape=(3, OUT_SIZE, OUT_SIZE),
                            resampling=Resampling.bilinear)

            # skips nodata tiles, i.e. everything outside the evaluation polygon
            if np.mean(tile) < 5:
                continue

            # ultralytics expects BGR channel order
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
gdf.to_file(f"{out_dir}/tree_detections_gtregion.geojson", driver="GeoJSON")
gdf.to_file(f"{out_dir}/tree_detections_gtregion.shp")
print(f"Wrote {len(gdf)} trees to {out_dir}/tree_detections_gtregion.geojson (+ .shp)")