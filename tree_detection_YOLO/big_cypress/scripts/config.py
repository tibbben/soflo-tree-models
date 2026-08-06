"""
config.py — single source of truth for a run's parameters.

Big Cypress variant. Differs from the campus version in two ways: the input is a
DIRECTORY of plot clips rather than a single survey raster, and there is no
evaluation-region crop (no usable ground truth exists for this site yet).

Usage from another script:
    from config import load, summary

Written by Ahsan and Claude.
"""

import yaml


def load(path):
    # read the yaml and compute the derived pixel values ONCE, here
    with open(path) as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]

    # tile overlap defaults to the crown radius, matching the campus pipeline
    d.setdefault("overlap_m", d["radius_m"])

    # The crown box in OUTPUT-tile pixels. NOT used for detection — the model predicts
    # its own boxes, and their size is fixed in its trained weights. Kept only so the
    # printed summary shows the apparent crown scale the model was trained at.
    d["crown_radius_out_px"] = int(d["radius_m"] / d["ground_m"] * d["out_size"])

    return cfg


def summary(cfg, stage):
    # printed by every stage — the fingerprint of what actually ran
    d = cfg["data"]
    print(f"[{stage}] config: {cfg['name']}")
    print(f"[{stage}] plot dir: {d['plot_dir']}")
    print(f"[{stage}] {d['ground_m']}m ground -> {d['out_size']}px tile | "
          f"reference crown radius {d['radius_m']}m = {d['crown_radius_out_px']}px | "
          f"overlap {d['overlap_m']}m")
