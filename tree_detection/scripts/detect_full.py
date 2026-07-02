"""
detect_full.py — Full-survey tree detection (the deliverable).

Tiles the entire survey window-by-window, runs the trained model on each tile,
converts pixel boxes to world coords (EPSG:32617), cross-tile NMS in world space,
exports one point per tree to GeoJSON + shapefile.

Run 16 config: the 5cm survey, tiled at 640px with 100px overlap, imgsz 1280. The
tiling MUST match how the weights were trained (CHIP_SIZE / OVERLAP / IMGSZ) or the
model sees tiles at the wrong scale.

Run from the project root:
    python scripts/detect_full.py <weights> [conf] [nms_iou]

Output: ./output/tree_detections.geojson  and  ./output/tree_detections.shp
"""

import sys
import os
import numpy as np
import torch
from torchvision.ops import nms
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
from ultralytics import YOLO

# --- Args (parameterized by sys.argv, not argparse) ---
if len(sys.argv) < 2:
    sys.exit("Usage: python detect_full.py <weights> [conf] [nms_iou]")
weights_path = sys.argv[1]
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
nms_iou = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

# 5cm survey — the SAME survey the Run 16 chips were cut from.
tiff_path = "./download/umgables_2025/umgables_2025_drone_survey_5cm.tif"
out_dir = "./output"
os.makedirs(out_dir, exist_ok=True)

# Tiling settings — MUST MATCH training (5cm, 640px chips, 100px overlap).
CHIP_SIZE = 640
OVERLAP = 100
IMGSZ = 1280
step = CHIP_SIZE - OVERLAP


def offsets(total, size, step):
    """Tile start offsets including a final clamped window so the far edge isn't dropped."""
    offs = list(range(0, total - size, step))
    if not offs or offs[-1] != total - size:
        offs.append(total - size)
    return offs


# Load the trained model once.
model = YOLO(weights_path)

world_boxes = []   # each: [minx, miny, maxx, maxy]
world_confs = []

with rasterio.open(tiff_path) as src:
    raster_crs = src.crs
    row_offs = offsets(src.height, CHIP_SIZE, step)
    col_offs = offsets(src.width, CHIP_SIZE, step)
    total_tiles = len(row_offs) * len(col_offs)
    print(f"Survey {src.width}x{src.height}px -> {total_tiles} tiles. Detecting...")

    done = 0
    for row_off in row_offs:
        for col_off in col_offs:
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total_tiles} tiles | {len(world_boxes)} detections so far")

            window = Window(col_off, row_off, CHIP_SIZE, CHIP_SIZE)
            tile = src.read(window=window)[:3]   # (3, H, W), RGB

            if np.mean(tile) < 5:
                continue

            # RGB -> BGR; ultralytics expects cv2-style BGR numpy arrays.
            tile_bgr = np.ascontiguousarray(np.transpose(tile, (1, 2, 0))[:, :, ::-1])

            results = model.predict(tile_bgr, imgsz=IMGSZ, conf=conf, verbose=False)
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes_px = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            win_t = src.window_transform(window)

            for (x1, y1, x2, y2), c in zip(boxes_px, confs):
                wx1, wy1 = win_t * (x1, y1)
                wx2, wy2 = win_t * (x2, y2)
                world_boxes.append([min(wx1, wx2), min(wy1, wy2),
                                    max(wx1, wx2), max(wy1, wy2)])
                world_confs.append(float(c))

print(f"Raw detections (with tile-overlap duplicates): {len(world_boxes)}")
if len(world_boxes) == 0:
    sys.exit("No detections — check the weights path and confidence threshold.")

# Cross-tile NMS in world space.
keep = nms(torch.tensor(world_boxes, dtype=torch.float32),
           torch.tensor(world_confs, dtype=torch.float32), nms_iou).numpy()
print(f"After cross-tile NMS: {len(keep)} unique trees")

points = [Point((world_boxes[i][0] + world_boxes[i][2]) / 2,
                (world_boxes[i][1] + world_boxes[i][3]) / 2) for i in keep]
out_confs = [world_confs[i] for i in keep]

gdf = gpd.GeoDataFrame({"confidence": out_confs}, geometry=points, crs=raster_crs)
geojson_path = f"{out_dir}/tree_detections.geojson"
shp_path = f"{out_dir}/tree_detections.shp"
gdf.to_file(geojson_path, driver="GeoJSON")
gdf.to_file(shp_path)

print(f"Wrote {len(gdf)} trees to:")
print(f"  {geojson_path}")
print(f"  {shp_path}")