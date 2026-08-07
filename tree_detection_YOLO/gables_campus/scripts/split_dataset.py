"""
split_dataset.py — Randomly split tiles into train/val sets (80/20).

Run from the project root AFTER tiling everything into train:
    python scripts/split_dataset.py

Moves 20% of the tiles in images/train + labels/train into images/val + labels/val,
keeping each image with its matching label. Fixed seed for reproducibility.

Written by Ahsan and Claude.
"""

import os
import random
import shutil

# Paths (relative, forward slashes, start with "./")
images_train = "./yolo_dataset/images/train"
labels_train = "./yolo_dataset/labels/train"
images_val = "./yolo_dataset/images/val"
labels_val = "./yolo_dataset/labels/val"
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

VAL_FRACTION = 0.2   # put 20% of tiles into validation
random.seed(42)      # fixed seed so the split is reproducible

# List all tiles currently in the train folder.
all_tiles = [f for f in os.listdir(images_train) if f.endswith(".png")]
random.shuffle(all_tiles)

# Take the first 20% as validation.
n_val = int(len(all_tiles) * VAL_FRACTION)
val_tiles = all_tiles[:n_val]

# Move each selected tile and its label from train -> val.
for tile in val_tiles:
    label = tile.replace(".png", ".txt")
    shutil.move(os.path.join(images_train, tile), os.path.join(images_val, tile))
    shutil.move(os.path.join(labels_train, label), os.path.join(labels_val, label))

print(f"Total tiles: {len(all_tiles)} | train: {len(all_tiles) - n_val} | val: {n_val}")
