"""
split_dataset.py — Randomly split chips into train/val sets (80/20).

Run from the gables_campus/ folder AFTER chipping everything into train:
    python scripts/split_dataset.py

Moves 20% of the chips in images/train + labels/train into images/val + labels/val,
keeping each image with its matching label. Uses a fixed seed for reproducibility.
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

VAL_FRACTION = 0.2   # put 20% of chips into validation
random.seed(42)      # fixed seed so the split is reproducible

# List all chips currently in the train folder.
all_chips = [f for f in os.listdir(images_train) if f.endswith(".png")]
random.shuffle(all_chips)

# Take the first 20% as validation.
n_val = int(len(all_chips) * VAL_FRACTION)
val_chips = all_chips[:n_val]

# Move each selected chip and its label from train -> val.
for chip in val_chips:
    label = chip.replace(".png", ".txt")
    shutil.move(os.path.join(images_train, chip), os.path.join(images_val, chip))
    shutil.move(os.path.join(labels_train, label), os.path.join(labels_val, label))

print(f"Total chips: {len(all_chips)} | train: {len(all_chips) - n_val} | val: {n_val}")