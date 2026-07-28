"""
chip_data.py — config-driven chipper: tiles a survey into labelled chips for training.

Emits constant-size chips covering a constant ground footprint at any source
resolution, so crowns are always the same pixel size and the ONLY variable between
runs is whatever the config changes.

Run from the project root:
    python scripts/chip_data.py ./configs/champion_5cm_5m.yaml
"""

import sys
import os
import random
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np
from PIL import Image

sys.path.insert(0, "./scripts")
from config import load, summary

random.seed(42)

if len(sys.argv) < 2:
    sys.exit("Usage: python scripts/chip_data.py <config.yaml>")
cfg = load(sys.argv[1])
summary(cfg, "chip")

d = cfg["data"]
OUT_SIZE = d["out_size"]
GROUND_M = d["ground_m"]
RADIUS_M = d["radius_m"]
OVERLAP_M = d["overlap_m"]
BG_RATIO = d["bg_ratio"]
crown_radius_out_px = d["crown_radius_out_px"]

split = "train"
images_out = f"./yolo_dataset/images/{split}"
labels_out = f"./yolo_dataset/labels/{split}"

# guard: stale chips from a previous run would silently contaminate this one
if os.path.isdir(images_out) and os.listdir(images_out):
    sys.exit(f"{images_out} is not empty — wipe it first:\n    rm -rf ./yolo_dataset")

os.makedirs(images_out, exist_ok=True)
os.makedirs(labels_out, exist_ok=True)

trees = gpd.read_file(d["trees"])

chip_count = 0
tree_chip_count = 0
bg_count = 0
box_count = 0


def offsets(total, size, step):
    # tile start positions, with a final offset flush to the raster edge
    offs = list(range(0, total - size, step))
    if not offs or offs[-1] != total - size:
        offs.append(total - size)
    return offs


with rasterio.open(d["survey"]) as src:
    if src.dtypes[0] != "uint8":
        sys.exit(f"Expected 8-bit RGB but got '{src.dtypes[0]}'")
    if trees.crs != src.crs:
        trees = trees.to_crs(src.crs)

    pixel_size = src.res[0]
    src_chip_px = int(round(GROUND_M / pixel_size))
    src_overlap_px = int(round(OVERLAP_M / pixel_size))
    step = src_chip_px - src_overlap_px

    print(f"source {pixel_size:.4f} m/px | reading {src_chip_px}px windows "
          f"-> emitting {OUT_SIZE}px chips")

    for row_off in offsets(src.height, src_chip_px, step):
        for col_off in offsets(src.width, src_chip_px, step):
            window = Window(col_off, row_off, src_chip_px, src_chip_px)

            # Read and resample to the constant output size. The finest survey is the
            # 5cm one, where a 32m window is exactly 640 source px and this is a no-op;
            # coarser surveys read fewer source px and get UPSAMPLED here, which is
            # exactly the detail difference the resolution experiments tested.
            # Bilinear is the right choice for upsampling. (Downsampling would want
            # area-averaging instead, to avoid aliasing.)
            chip = src.read([1, 2, 3], window=window,
                            out_shape=(3, OUT_SIZE, OUT_SIZE),
                            resampling=Resampling.bilinear)

            # skip near-empty tiles (nodata / outside the survey footprint)
            if np.mean(chip) < 5:
                continue

            win_t = src.window_transform(window)
            left = win_t.c
            top = win_t.f
            right = left + src_chip_px * pixel_size
            bottom = top - src_chip_px * pixel_size

            chip_trees = trees.cx[left:right, bottom:top]
            has_trees = len(chip_trees) > 0

            # keep a limited number of tree-free chips as negative examples
            if not has_trees:
                if tree_chip_count > 0 and bg_count < tree_chip_count // BG_RATIO:
                    if random.random() >= 0.3:
                        continue
                else:
                    continue

            yolo_labels = []
            if has_trees:
                for _, tree in chip_trees.iterrows():
                    # world -> output-chip px (scale by ground/out_size, since the
                    # window was resampled regardless of source pixel size)
                    px = (tree.geometry.x - left) / GROUND_M * OUT_SIZE
                    py = (top - tree.geometry.y) / GROUND_M * OUT_SIZE

                    # fixed crown box, clipped to the chip edges
                    xmin = max(0, px - crown_radius_out_px)
                    ymin = max(0, py - crown_radius_out_px)
                    xmax = min(OUT_SIZE, px + crown_radius_out_px)
                    ymax = min(OUT_SIZE, py + crown_radius_out_px)

                    # YOLO format: class, normalised centre x/y, normalised w/h
                    cx = (xmin + xmax) / 2 / OUT_SIZE
                    cy = (ymin + ymax) / 2 / OUT_SIZE
                    w = (xmax - xmin) / OUT_SIZE
                    h = (ymax - ymin) / OUT_SIZE

                    yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    box_count += 1
                tree_chip_count += 1
            else:
                bg_count += 1

            name = f"chip_{chip_count:05d}"
            img = Image.fromarray(chip.transpose(1, 2, 0).astype(np.uint8))
            img.save(f"{images_out}/{name}.png")
            with open(f"{labels_out}/{name}.txt", "w") as f:
                f.write("\n".join(yolo_labels))
            chip_count += 1

print(f"Chips: {chip_count} ({tree_chip_count} with trees, {bg_count} background)")
print(f"Tree boxes: {box_count} (all {RADIUS_M}m = {crown_radius_out_px}px)")