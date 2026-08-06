"""
config.py — single source of truth for a run's parameters.

Every pipeline stage (tile / train / detect) loads the SAME yaml, so the config drift
that used to require editing the tiler and the detectors in lockstep is now
structurally impossible.

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

    # tile overlap defaults to the crown radius (what every run so far has used)
    d.setdefault("overlap_m", d["radius_m"])

    # the crown box in OUTPUT-tile pixels — constant at any source resolution
    # 5m -> 100px, 7m -> 140px, 3m -> 60px (with ground_m 32, out_size 640)
    d["crown_radius_out_px"] = int(d["radius_m"] / d["ground_m"] * d["out_size"])

    return cfg


def summary(cfg, stage):
    # printed by every stage — the fingerprint of what actually ran
    d = cfg["data"]
    print(f"[{stage}] config: {cfg['name']}")
    print(f"[{stage}] survey: {d['survey']}")
    print(f"[{stage}] gtregion: {d['survey_gtregion']}")
    print(f"[{stage}] {d['ground_m']}m ground -> {d['out_size']}px tile | "
          f"radius {d['radius_m']}m = {d['crown_radius_out_px']}px | "
          f"overlap {d['overlap_m']}m")
