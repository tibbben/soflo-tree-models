"""
Talk figures — UM Data Science department talk, September 2026.

Regenerates the two figures that were built for the talk and exist nowhere else in
the pipeline. Everything else in the deck comes from the committed `reports/` PNGs.

  1. figures/audit_verdicts.jpg   four audited "false positives" from the Gables run
                                  (s10), labelled with the verdict I gave each one
                                  by eye in QGIS.
  2. figures/bcnp_three_panel.jpg one 30 x 30 m window of the held-out BCNP plot
                                  plot_9_3, three ways: bare imagery, the census
                                  trees inside it, and the s16 YOLO detections for
                                  the same ground.

Inputs (none of them are in git):
  outputs/audit/fp_crops/fp_XXX.png             <- src/s10_audit.py (build step)
  outputs/audit/fp_audit_completed.csv          <- committed; my audit tags
  outputs/bcnp_yolo/test_raw_predictions.npz    <- src/s17_bcnp_size_eval.py cache
  $BCNP/plot_9_3.tif, RP.Plot_census_data.2025.csv, tree_plots_polygons.geojson

BCNP data root defaults to ~/Downloads/bcnp (same as s13-s18); override with
the BCNP_DIR environment variable.

Run:  .venv/bin/python talks/2026-09-ds-department/make_figures.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
TD = HERE.parents[1]                      # treedetect/
OUT = HERE / "figures"
BCNP = Path(os.environ.get("BCNP_DIR", "~/Downloads/bcnp")).expanduser()

PLOT = "plot_9_3"                          # held-out test plot
UNIT, UNITPLOT = 9, 3
WINDOW_M = 30.0                            # size of the zoom window
CONF = 0.02                                # s16 operating point
NMS_M = 0.8                                # s16 peak_min_dist_m

# fp_id -> (label, colour). Verdicts are mine, from outputs/audit/fp_audit_completed.csv.
AUDIT_PANELS = [
    (2,  "Real tree\nnot in the inventory", "#1b7837"),
    (4,  "Real tree\nnot in the inventory", "#1b7837"),
    (7,  "Real trees\none box over several crowns", "#1b7837"),
    (82, "A real error\nbush, not a tree", "#b2182b"),
]


def audit_figure():
    crops = TD / "outputs" / "audit" / "fp_crops"
    missing = [i for i, _, _ in AUDIT_PANELS if not (crops / f"fp_{i:03d}.png").exists()]
    if missing:
        print(f"[audit] missing crops {missing} — run `python src/s10_audit.py` first; skipping")
        return
    fig, axes = plt.subplots(1, len(AUDIT_PANELS), figsize=(14.5, 4.4))
    for ax, (fp_id, label, colour) in zip(axes, AUDIT_PANELS):
        ax.imshow(Image.open(crops / f"fp_{fp_id:03d}.png").convert("RGB"))
        ax.axis("off")
        ax.set_title(label, fontsize=14, color=colour, fontweight="bold")
    fig.suptitle('The audit: for every red box the model drew, I opened the imagery '
                 'and asked "is there a real tree in here?"', fontsize=15, y=1.04)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "audit_verdicts.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'audit_verdicts.jpg'}")


def nms(points, scores, min_dist_m):
    """Greedy NMS, same rule as s16: keep the highest score, drop neighbours."""
    from scipy.spatial import cKDTree
    order = np.argsort(-scores)
    pts, sc = points[order], scores[order]
    keep = np.ones(len(pts), bool)
    tree = cKDTree(pts)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(pts[i], min_dist_m):
            if j > i and keep[j]:
                keep[j] = False
    return pts[keep]


def bcnp_figure():
    import rasterio
    from rasterio.windows import from_bounds
    import geopandas as gpd

    tif = BCNP / f"{PLOT}.tif"
    census_csv = BCNP / "RP.Plot_census_data.2025.csv"
    polys_gj = BCNP / "tree_plots_polygons.geojson"
    preds_npz = TD / "outputs" / "bcnp_yolo" / "test_raw_predictions.npz"
    for p in (tif, census_csv, polys_gj, preds_npz):
        if not p.exists():
            print(f"[bcnp] missing {p} — skipping")
            return

    poly = gpd.read_file(polys_gj).to_crs(32617)
    poly = poly[poly.Name == PLOT].geometry.iloc[0]
    swx, swy = poly.bounds[0], poly.bounds[1]

    census = pd.read_csv(census_csv)
    sub = census[(census.UNIT == UNIT) & (census.UNITPLOT == UNITPLOT) & (census.ALIVE == 1)]
    sub = sub.dropna(subset=["XCOORD", "YCOORD"])
    cx = swx + pd.to_numeric(sub.XCOORD, errors="coerce").values
    cy = swy + pd.to_numeric(sub.YCOORD, errors="coerce").values

    z = np.load(preds_npz)
    xy, sc = z[f"{PLOT}_xy"], z[f"{PLOT}_sc"]
    keep = sc >= CONF
    det = nms(xy[keep], sc[keep], NMS_M)
    print(f"[bcnp] census alive {len(cx)}, detections after NMS {len(det)}")

    # centre window on the middle of the 100 m plot
    x0, x1 = swx + 50 - WINDOW_M / 2, swx + 50 + WINDOW_M / 2
    y0, y1 = swy + 50 - WINDOW_M / 2, swy + 50 + WINDOW_M / 2
    with rasterio.open(tif) as src:
        win = from_bounds(x0, y0, x1, y1, src.transform)
        img = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0))
        wt = src.window_transform(win)

    def to_px(x, y):
        return (x - wt.c) / wt.a, (y - wt.f) / wt.e

    mc = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    md = (det[:, 0] >= x0) & (det[:, 0] <= x1) & (det[:, 1] >= y0) & (det[:, 1] <= y1)
    h, w = img.shape[:2]

    panels = [
        (f"The drone photo\n{WINDOW_M:.0f} m x {WINDOW_M:.0f} m of Big Cypress", None, None),
        (f"What the ground survey says\n{int(mc.sum())} surveyed trees in this window",
         (cx[mc], cy[mc]), "#39ff14"),
        (f"What the model returns\n{int(md.sum())} detections in the same window",
         (det[md, 0], det[md, 1]), "#d73027"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    for ax, (title, pts, colour) in zip(axes, panels):
        ax.imshow(img)
        if pts is not None:
            px, py = to_px(pts[0], pts[1])
            if colour == "#39ff14":
                ax.scatter(px, py, s=70, marker="+", c=colour, linewidths=1.6)
            else:
                ax.scatter(px, py, s=60, facecolors="none", edgecolors=colour, linewidths=1.4)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
        ax.set_title(title, fontsize=14)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "bcnp_three_panel.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'bcnp_three_panel.jpg'}")


if __name__ == "__main__":
    audit_figure()
    bcnp_figure()
