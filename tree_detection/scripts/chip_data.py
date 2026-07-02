"""
chip_data.py — Tile a GeoTIFF into chips for YOLOv8, and write matching YOLO-format
bounding-box labels from the tree points.

Run 16 config: the 5cm survey, tiled at 640px with 100px overlap. Every tree gets the
SAME fixed 5.0 m crown box (uniform sizing; variable sizing was tested and lost). At
0.05 m/px a 5.0 m radius is int(5.0/0.05) = 100 px (a 200 px box) inside a 640 px tile.

Run from the project root:
    python scripts/chip_data.py full         # whole survey -> images/train
    python scripts/chip_data.py training     # training area -> images/train
    python scripts/chip_data.py validation   # validation area -> images/val
"""

import sys
import os
import random
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image

# Fixed seed so background-chip sampling is reproducible across re-chips.
random.seed(42)

# Mode from the command line: "full", "training", or "validation".
if len(sys.argv) < 2:
    sys.exit("Usage: python chip_data.py <mode>   (full | training | validation)")
mode = sys.argv[1]

# Pick inputs and the output split based on the mode.
if mode == "full":
    tiff_path = "./download/umgables_2025/umgables_2025_drone_survey_5cm.tif"  # 5cm survey
    trees_path = "./download/um_gables_trees.geojson"
    split = "train"   # everything goes to train; split_dataset.py carves out val
else:
    tiff_path = f"./{mode}_area.tif"
    trees_path = f"./{mode}_trees.geojson"
    split = {"training": "train", "validation": "val"}[mode]

# Settings — 640px tiles on the 5cm survey.
CHIP_SIZE = 640         # chip size in pixels (square)
OVERLAP = 100           # overlap in pixels
RADIUS_M = 5.0          # fixed crown radius for EVERY tree (uniform sizing)
BG_RATIO = 3            # keep 1 background chip (no trees) for every N tree chips


def offsets(total, size, step):
    """Chip start offsets along one axis, INCLUDING a final clamped window so the
    far edge isn't dropped (plain range() stops short of it)."""
    offs = list(range(0, total - size, step))
    if not offs or offs[-1] != total - size:
        offs.append(total - size)
    return offs


# Output folders (relative, forward slashes, start with "./")
images_out = f"./yolo_dataset/images/{split}"
labels_out = f"./yolo_dataset/labels/{split}"
os.makedirs(images_out, exist_ok=True)
os.makedirs(labels_out, exist_ok=True)

# Load tree labels (CRS aligned to the raster once it's open).
trees = gpd.read_file(trees_path)

chip_count = 0
tree_chip_count = 0
bg_count = 0
box_count = 0   # total tree boxes written (all at the fixed 5m radius)

with rasterio.open(tiff_path) as src:
    # Guard: we cast to uint8 for PNG; a uint16/float raster would be corrupted.
    if src.dtypes[0] != "uint8":
        sys.exit(f"Expected 8-bit RGB but got '{src.dtypes[0]}' — convert to uint8 first.")

    # Guard: align label CRS to the raster, else .cx matches nothing.
    if trees.crs != src.crs:
        trees = trees.to_crs(src.crs)

    pixel_size = src.res[0]     # meters per pixel (0.05 for the 5cm survey)
    step = CHIP_SIZE - OVERLAP  # stride

    # Crown radius is the same for every tree, so compute it once (px).
    crown_radius_px = int(RADIUS_M / pixel_size)   # 5.0 / 0.05 = 100 px

    # Slide the window across the raster (edges included via offsets()).
    for row_off in offsets(src.height, CHIP_SIZE, step):
        for col_off in offsets(src.width, CHIP_SIZE, step):
            window = Window(col_off, row_off, CHIP_SIZE, CHIP_SIZE)
            chip = src.read(window=window)[:3]         # (3, H, W) — RGB only

            # Skip near-empty chips (nodata / black border around the survey).
            if np.mean(chip) < 5:
                continue

            # Window pixel coords -> world (UTM) coords for this chip.
            win_t = src.window_transform(window)
            left = win_t.c
            top = win_t.f
            right = left + CHIP_SIZE * pixel_size
            bottom = top - CHIP_SIZE * pixel_size

            # Tree points inside this chip (.cx takes [xmin:xmax, ymin:ymax]).
            chip_trees = trees.cx[left:right, bottom:top]
            has_trees = len(chip_trees) > 0

            # Background chips: keep ~1 per BG_RATIO tree chips, randomly sampled.
            if not has_trees:
                if tree_chip_count > 0 and bg_count < tree_chip_count // BG_RATIO:
                    if random.random() >= 0.3:
                        continue   # thin out to spread backgrounds across the area
                else:
                    continue

            # Build YOLO label lines for tree chips; background chips get none.
            yolo_labels = []
            if has_trees:
                for _, tree in chip_trees.iterrows():
                    # World coords -> chip pixel coords.
                    px = (tree.geometry.x - left) / pixel_size
                    py = (top - tree.geometry.y) / pixel_size

                    # Square box at the fixed crown radius, clamped to chip bounds.
                    xmin = max(0, px - crown_radius_px)
                    ymin = max(0, py - crown_radius_px)
                    xmax = min(CHIP_SIZE, px + crown_radius_px)
                    ymax = min(CHIP_SIZE, py + crown_radius_px)

                    # Pixel box -> YOLO normalized (center x, center y, w, h).
                    cx = (xmin + xmax) / 2 / CHIP_SIZE
                    cy = (ymin + ymax) / 2 / CHIP_SIZE
                    w = (xmax - xmin) / CHIP_SIZE
                    h = (ymax - ymin) / CHIP_SIZE

                    yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")  # class 0 = Tree
                    box_count += 1

                tree_chip_count += 1
            else:
                bg_count += 1

            # Save the chip image (RGB) and its label file.
            name = f"chip_{chip_count:05d}"
            img = Image.fromarray(chip.transpose(1, 2, 0).astype(np.uint8))
            img.save(f"{images_out}/{name}.png")
            with open(f"{labels_out}/{name}.txt", "w") as f:
                f.write("\n".join(yolo_labels))  # empty for background chips

            chip_count += 1

print(f"Chips created in images/{split}: {chip_count} ({tree_chip_count} with trees, {bg_count} background)")
print(f"Tree boxes: {box_count} (all at fixed {RADIUS_M}m radius = {crown_radius_px}px)")