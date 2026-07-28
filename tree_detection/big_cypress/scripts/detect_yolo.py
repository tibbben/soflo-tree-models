"""
detect_yolo.py — Run the campus-trained YOLO champion across the Big Cypress plot
clips. One point per detected tree crown, written per plot plus a merged layer.

There is no usable ground truth for this site yet, so there is NO benchmarking step and
no evidence-based way to pick a confidence threshold. Detections are produced at a low
floor and the 'confidence' attribute is kept on every point, so thresholds can be
explored visually in QGIS by filtering on that attribute rather than re-running.

NOTE ON DOMAIN SHIFT: the weights were trained on a managed campus (pruned ornamentals
and palms on lawn/pavement) at 5cm. Big Cypress is closed-canopy wetland forest. Tiles
are cut to the same GROUND footprint the model was trained at, so crowns appear at a
similar pixel scale regardless of this survey's resolution — that is the best available
transfer, but performance is expected to drop and the output should be treated as
provisional.

Run from the big_cypress project root:
    python scripts/detect_yolo.py <config.yaml> <weights> [conf]

Output: ./output/yolo/<plot>.geojson  plus  ./output/yolo_all_plots.geojson
"""

import sys
import os
import glob
import numpy as np
import torch
from torchvision.ops import nms
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from ultralytics import YOLO

sys.path.insert(0, "./scripts")
from config import load, summary

if len(sys.argv) < 3:
    sys.exit("Usage: python scripts/detect_yolo.py <config.yaml> <weights> [conf]")
cfg = load(sys.argv[1])
summary(cfg, "detect_yolo")
weights_path = sys.argv[2]
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

d = cfg["data"]
OUT_SIZE = d["out_size"]
GROUND_M = d["ground_m"]
OVERLAP_M = d["overlap_m"]
IMGSZ = cfg["detect"]["imgsz"]
NMS_IOU = cfg["detect"]["nms_iou"]

out_dir = "./output/yolo"
os.makedirs(out_dir, exist_ok=True)

plot_files = sorted(glob.glob(f"{d['plot_dir']}/*.tif"))
if not plot_files:
    sys.exit(f"No .tif files found in {d['plot_dir']}")
print(f"Found {len(plot_files)} plot clips. Confidence floor {conf}.\n")


def offsets(total, size, step):
    # tile start positions, with a final offset flush to the raster edge
    offs = list(range(0, total - size, step))
    if not offs or offs[-1] != total - size:
        offs.append(total - size)
    return offs


def to_uint8(arr, dtype_name):
    # This site's clips are already uint8 and pass through untouched. Other surveys
    # are often uint16, so scale rather than refusing to run. Note the scaling is
    # per-tile, which would make brightness inconsistent across tiles on non-uint8
    # imagery — revisit if a 16-bit survey is ever run through this.
    if dtype_name == "uint8":
        return arr.astype(np.uint8)
    a = arr.astype(np.float32)
    hi = a.max()
    if hi <= 0:
        return np.zeros_like(a, dtype=np.uint8)
    return np.clip(a / hi * 255.0, 0, 255).astype(np.uint8)


model = YOLO(weights_path)
all_frames = []

for pi, plot_path in enumerate(plot_files, start=1):
    plot_name = os.path.splitext(os.path.basename(plot_path))[0]
    world_boxes = []
    world_confs = []
    skipped = 0

    with rasterio.open(plot_path) as src:
        raster_crs = src.crs
        pixel_size = src.res[0]
        dtype_name = src.dtypes[0]

        # tile by GROUND distance, so crowns appear at the scale the model expects
        src_chip_px = int(round(GROUND_M / pixel_size))
        src_overlap_px = int(round(OVERLAP_M / pixel_size))
        step = max(1, src_chip_px - src_overlap_px)

        if src.count < 3:
            print(f"[{pi}/{len(plot_files)}] {plot_name}: only {src.count} band(s), skipping")
            continue

        row_offs = offsets(src.height, src_chip_px, step)
        col_offs = offsets(src.width, src_chip_px, step)
        total_tiles = len(row_offs) * len(col_offs)

        print(f"[{pi}/{len(plot_files)}] {plot_name}: {src.width}x{src.height}px @ "
              f"{pixel_size:.4f}m/px ({dtype_name}, {src.count} bands) -> "
              f"{total_tiles} tiles of {src_chip_px}px")

        for row_off in row_offs:
            for col_off in col_offs:
                window = Window(col_off, row_off, src_chip_px, src_chip_px)

                # Read the first three bands only; the 4th band is alpha, not NIR.
                # AVERAGE resampling, not bilinear: these windows are DOWNsampled to
                # out_size (1893 source px -> 640 here), and averaging is the
                # anti-aliased choice when shrinking. Bilinear samples sparse points
                # and aliases, manufacturing spurious edge texture across continuous
                # canopy. The campus chipper uses bilinear because it UPsamples.
                tile = src.read([1, 2, 3], window=window,
                                out_shape=(3, OUT_SIZE, OUT_SIZE),
                                resampling=Resampling.average)
                tile = to_uint8(tile, dtype_name)

                # skip near-empty tiles (nodata / outside the clip footprint)
                if np.mean(tile) < 5:
                    skipped += 1
                    continue

                # ultralytics expects BGR channel order
                tile_bgr = np.ascontiguousarray(
                    np.transpose(tile, (1, 2, 0))[:, :, ::-1])
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

    if len(world_boxes) == 0:
        print(f"    no detections ({skipped} empty tiles skipped)")
        continue

    # cross-tile NMS: the same tree seen in two overlapping tiles becomes one detection
    keep = nms(torch.tensor(world_boxes, dtype=torch.float32),
               torch.tensor(world_confs, dtype=torch.float32), NMS_IOU).numpy()

    # one POINT per tree, at the box centroid
    points = [Point((world_boxes[i][0] + world_boxes[i][2]) / 2,
                    (world_boxes[i][1] + world_boxes[i][3]) / 2) for i in keep]
    out_confs = [world_confs[i] for i in keep]

    gdf = gpd.GeoDataFrame({"confidence": out_confs, "plot": plot_name},
                           geometry=points, crs=raster_crs)
    gdf.to_file(f"{out_dir}/{plot_name}.geojson", driver="GeoJSON")
    all_frames.append(gdf)
    print(f"    {len(world_boxes)} raw -> {len(gdf)} after NMS "
          f"({skipped} empty tiles skipped)")

# merged layer across all plots, for loading as a single QGIS layer
if all_frames:
    merged = gpd.GeoDataFrame(pd.concat(all_frames, ignore_index=True),
                              crs=all_frames[0].crs)
    merged.to_file("./output/yolo_all_plots.geojson", driver="GeoJSON")
    print(f"\nWrote {len(merged)} total detections across {len(all_frames)} plots")
    print("  per-plot: ./output/yolo/<plot>.geojson")
    print("  merged:   ./output/yolo_all_plots.geojson")
else:
    print("\nNo detections in any plot.")