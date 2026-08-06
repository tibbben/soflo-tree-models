"""
detect_deepforest.py — Run the PRETRAINED DeepForest release model across the Big
Cypress plot clips. One point per detected tree crown, per plot plus a merged layer.

Uses the released 'weecology/deepforest-tree' weights, NOT the campus fine-tunes. The
release model was pretrained on NEON forest canopy crowns, which is a much closer
domain to Big Cypress wetland forest than a managed campus is. The campus fine-tunes
specialised the model AWAY from natural forest and should not be used here.

On campus, DeepForest lost to YOLO26 (F1 0.542 vs 0.660) because the domain suited
YOLO. On visual inspection that ranking appears to invert at this site — most clearly
on bare, pale crowns, which this model detects and the campus champion misses entirely.
With no ground truth there is no way to confirm that numerically, so both models are run
and compared visually.

Requires the separate 'deepforest' environment (deepforest==1.5.2, albumentations<2.0).

Run from the big_cypress project root:
    python scripts/detect_deepforest.py <config.yaml> [score_thresh]

Output: ./output/deepforest/<plot>.geojson  plus  ./output/deepforest_all_plots.geojson

Written by Ahsan and Claude.
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
from deepforest import main as df_main

sys.path.insert(0, "./scripts")
from config import load, summary

if len(sys.argv) < 2:
    sys.exit("Usage: python scripts/detect_deepforest.py <config.yaml> [score_thresh]")
cfg = load(sys.argv[1])
summary(cfg, "detect_deepforest")
score_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

d = cfg["data"]
OUT_SIZE = d["out_size"]
GROUND_M = d["ground_m"]
OVERLAP_M = d["overlap_m"]
NMS_IOU = cfg["detect"]["nms_iou"]

out_dir = "./output/deepforest"
os.makedirs(out_dir, exist_ok=True)

plot_files = sorted(glob.glob(f"{d['plot_dir']}/*.tif"))
if not plot_files:
    sys.exit(f"No .tif files found in {d['plot_dir']}")
print(f"Found {len(plot_files)} plot clips. Score threshold {score_thresh}.\n")


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


# load the RELEASE weights (not a campus fine-tune)
model = df_main.deepforest()
model.load_model("weecology/deepforest-tree")
# score_thresh is set on both the config dict and the underlying model, because
# which one takes effect varies across deepforest versions
model.config["score_thresh"] = score_thresh
try:
    model.model.score_thresh = score_thresh
except AttributeError:
    pass
if torch.cuda.is_available():
    model.to("cuda")
model.eval()

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
        src_tile_px = int(round(GROUND_M / pixel_size))
        src_overlap_px = int(round(OVERLAP_M / pixel_size))
        step = max(1, src_tile_px - src_overlap_px)

        if src.count < 3:
            print(f"[{pi}/{len(plot_files)}] {plot_name}: only {src.count} band(s), skipping")
            continue

        row_offs = offsets(src.height, src_tile_px, step)
        col_offs = offsets(src.width, src_tile_px, step)
        total_tiles = len(row_offs) * len(col_offs)

        print(f"[{pi}/{len(plot_files)}] {plot_name}: {src.width}x{src.height}px @ "
              f"{pixel_size:.4f}m/px ({dtype_name}, {src.count} bands) -> "
              f"{total_tiles} tiles of {src_tile_px}px")

        for row_off in row_offs:
            for col_off in col_offs:
                window = Window(col_off, row_off, src_tile_px, src_tile_px)

                # Read the first three bands only; the 4th band is alpha, not NIR.
                # AVERAGE resampling, not bilinear: these windows are DOWNsampled to
                # out_size (2367 source px -> 400 here), and averaging is the
                # anti-aliased choice when shrinking. Bilinear samples sparse points
                # and aliases, manufacturing spurious edge texture across continuous
                # canopy.
                tile = src.read([1, 2, 3], window=window,
                                out_shape=(3, OUT_SIZE, OUT_SIZE),
                                resampling=Resampling.average)
                tile = to_uint8(tile, dtype_name)

                # skip near-empty tiles (nodata / outside the clip footprint)
                if np.mean(tile) < 5:
                    skipped += 1
                    continue

                # DeepForest expects RGB (no BGR flip, unlike ultralytics)
                tile_rgb = np.ascontiguousarray(np.transpose(tile, (1, 2, 0)))

                try:
                    preds = model.predict_image(image=tile_rgb.astype(np.float32))
                except Exception as e:
                    print(f"    tile predict failed: {e}")
                    continue

                if preds is None or len(preds) == 0:
                    continue

                win_t = src.window_transform(window)
                left = win_t.c
                top = win_t.f

                for _, row in preds.iterrows():
                    x1, y1 = float(row["xmin"]), float(row["ymin"])
                    x2, y2 = float(row["xmax"]), float(row["ymax"])
                    c = float(row["score"])

                    # output-tile px -> world coordinates
                    wx1 = left + x1 / OUT_SIZE * GROUND_M
                    wx2 = left + x2 / OUT_SIZE * GROUND_M
                    wy1 = top - y1 / OUT_SIZE * GROUND_M
                    wy2 = top - y2 / OUT_SIZE * GROUND_M
                    world_boxes.append([min(wx1, wx2), min(wy1, wy2),
                                        max(wx1, wx2), max(wy1, wy2)])
                    # stored as 'confidence' for consistency with the YOLO output,
                    # though the two models' score scales are NOT comparable
                    world_confs.append(c)

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
    merged.to_file("./output/deepforest_all_plots.geojson", driver="GeoJSON")
    print(f"\nWrote {len(merged)} total detections across {len(all_frames)} plots")
    print("  per-plot: ./output/deepforest/<plot>.geojson")
    print("  merged:   ./output/deepforest_all_plots.geojson")
else:
    print("\nNo detections in any plot.")
