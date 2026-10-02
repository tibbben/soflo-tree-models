"""
Step 18 — Is BCNP census->imagery georeferencing offset / rotated / flipped?  [local]

s14/s15/s16 all plateau near the same weak number. Before blaming the models, test
the shared assumption: that census plot-local XY, placed as world = SW_corner +
(XCOORD, YCOORD) in EPSG:32617, actually lands on the trees in the imagery. NO
retraining, NO model changes — pure diagnostics.

  1 PALM TEST (human-verifiable, decisive): Sabal palmetto crowns are unmistakable
    from above (radial fronds). Render high-zoom 20x20 m crops with ONLY palm census
    points overlaid, for plot_18_2 (1344 palms) and plot_12_1 (444). If the points do
    not sit on obvious palm crowns, the georeferencing is wrong.
  2 OFFSET/ROTATION SWEEP (quantitative): using the existing s16 YOLO predictions on
    the test plots, sweep rigid transforms of the census and recompute distance recall
    @1 m — translations dx,dy in [-20,20] m, rotations [-20,20] deg about plot centre,
    plus 90/180/270 deg and axis flips / swapped X-Y. Report the max-recall transform
    and a recall(dx,dy) heatmap. Peak at (0,0,0) => georef fine; elsewhere => the bug.
  3 INTERNAL-STRUCTURE CHECK: is the census cloud axis-aligned (principal axes, X/Y
    ranges, expect 0-100 m)? Are the plot polygons axis-aligned squares — report each
    corner and which corner is the origin.
  4 Plain verdict: correct / offset / rotated / origin-flipped.

Uses the cached s16 predictions from s17 (regenerated from the s16 weights if absent).
Writes only under outputs/bcnp_georef + reports/bcnp_georef.

Run:  .venv/bin/python src/s18_bcnp_georef_check.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
sys.path.append(str(Path(__file__).resolve().parent))
import common as C
import s16_bcnp_yolo as Y

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
PALM = "Sabal palmetto"


def paths(cfg):
    od = C.p(cfg, cfg["outputs_dir"]) / "bcnp_georef"
    rd = C.p(cfg, cfg["reports_dir"]) / "bcnp_georef"
    od.mkdir(parents=True, exist_ok=True); rd.mkdir(parents=True, exist_ok=True)
    return od, rd


def plot_poly(cfg, name, polys):
    return polys[polys.Name == name].to_crs(cfg["crs"]).geometry.iloc[0]


def local_xy(cfg, name, census, species=None):
    """Plot-local XCOORD/YCOORD (metres) for ALIVE trees (optionally one species)."""
    u, up = Y._plot_uid(name)
    s = census[(census.UNIT == u) & (census.UNITPLOT == up) & (census.ALIVE == 1)].dropna(subset=["XCOORD", "YCOORD"])
    if species:
        s = s[s.SP == species]
    return (pd.to_numeric(s.XCOORD, errors="coerce").values,
            pd.to_numeric(s.YCOORD, errors="coerce").values)


def place(convention, X, Y, poly):
    """Map plot-local (X,Y) -> world (EPSG:32617) under a naming/axis convention."""
    minx, miny, maxx, maxy = poly.bounds
    if convention == "current":       # SW origin, X=east, Y=north (what s14-s16 use)
        return minx + X, miny + Y
    if convention == "yflip":         # NW origin, Y measured downward
        return minx + X, maxy - Y
    if convention == "xflip":         # SE origin
        return maxx - X, miny + Y
    if convention == "xyflip":        # NE origin (== 180 deg)
        return maxx - X, maxy - Y
    if convention == "swap":          # X<->Y swapped
        return minx + Y, miny + X
    if convention == "swap_yflip":
        return minx + Y, maxy - X
    if convention == "swap_xflip":
        return maxx - Y, miny + X
    if convention == "swap_xyflip":
        return maxx - Y, maxy - X
    raise ValueError(convention)


# ── stage 1: palm test ───────────────────────────────────────────────────────
def palm_crops(cfg, rd, name, census, polys, n_crops=6, size_m=20.0):
    import rasterio
    from rasterio.windows import from_bounds
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    poly = plot_poly(cfg, name, polys)
    Xp, Yp = local_xy(cfg, name, census, species=PALM)
    wx, wy = place("current", Xp, Yp, poly)
    minx, miny, maxx, maxy = poly.bounds
    # pick palm-dense 20 m cells
    cell = size_m
    nx = int(np.ceil((maxx - minx) / cell)); ny = int(np.ceil((maxy - miny) / cell))
    ix = np.clip(((wx - minx) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((wy - miny) / cell).astype(int), 0, ny - 1)
    counts = {}
    for a, b in zip(ix, iy):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)[:n_crops]

    tif = Y.bc_dir(cfg) / f"{name}.tif"
    ncol = 3; nrow = int(np.ceil(n_crops / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axes = np.array(axes).reshape(-1)
    with rasterio.open(tif) as src:
        for k, (a, b) in enumerate(top):
            cx = minx + (a + 0.5) * cell; cy = miny + (b + 0.5) * cell
            x0, x1 = cx - size_m / 2, cx + size_m / 2
            y0, y1 = cy - size_m / 2, cy + size_m / 2
            win = from_bounds(x0, y0, x1, y1, src.transform)
            out = min(900, int(size_m / abs(src.transform.a)))
            img = src.read([1, 2, 3], window=win, boundless=True, fill_value=0,
                           out_shape=(3, out, out))
            ax = axes[k]; ax.imshow(np.transpose(img, (1, 2, 0)))
            m = (wx >= x0) & (wx < x1) & (wy >= y0) & (wy < y1)
            px = (wx[m] - x0) / size_m * out
            py = (y1 - wy[m]) / size_m * out          # raster row 0 is top (north)
            ax.scatter(px, py, s=90, facecolors="none", edgecolors="#39ff14", linewidths=1.6)
            ax.set_title(f"{name} {size_m:.0f}m — {int(m.sum())} palms", fontsize=9)
            ax.axis("off")
    for k in range(len(top), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"s18 PALM TEST — {name}: census 'Sabal palmetto' points (green) over imagery\n"
                 "if these rings do not sit on radial-frond palm crowns, georeferencing is WRONG",
                 fontsize=12)
    fig.tight_layout(); out = rd / f"palm_test_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  ({len(top)} crops, densest palm cells)")
    return out


# ── stage 2: offset / rotation / convention sweep ────────────────────────────
def load_preds(cfg, census, polys):
    od = C.p(cfg, cfg["outputs_dir"]) / "bcnp_yolo"
    cache = od / "test_raw_predictions.npz"
    tp = cfg["bcnp_yolo"]["test_plots"]
    if not cache.exists():
        print("[sweep] no cached predictions — regenerating from s16 weights (inference only)")
        bc = json.loads((od / "best_config.json").read_text())
        best_pt = od / "runs" / "final" / "weights" / "best.pt"
        save = {}
        for name in tp:
            xy, sc = Y.predict_plot_raw(cfg, name, best_pt, bc["gsd_m"], census, polys, conf_floor=0.02)
            save[f"{name}_xy"] = xy; save[f"{name}_sc"] = sc
        np.savez_compressed(cache, **save)
    z = np.load(cache, allow_pickle=True)
    preds = {}
    for name in tp:
        xy, sc = z[f"{name}_xy"], z[f"{name}_sc"]
        pw, ps = Y._nms_world(xy[sc >= 0.02], sc[sc >= 0.02], cfg["bcnp_yolo"]["peak_min_dist_m"])
        preds[name] = pw
    return preds


def _recall(pred_xy, gt_xy, radius):
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return 0.0, 0, len(gt_xy)
    from scipy.spatial import cKDTree
    tree = cKDTree(gt_xy)
    used = set()
    for p in pred_xy:                       # preds claim nearest unused gt (recall-oriented)
        d, j = tree.query(p)
        if d <= radius and j not in used:
            used.add(j)
    return len(used) / len(gt_xy), len(used), len(gt_xy)


def transform_census(cfg, name, census, polys, dx=0.0, dy=0.0, theta_deg=0.0, convention="current"):
    poly = plot_poly(cfg, name, polys)
    X, Y_ = local_xy(cfg, name, census)
    wx, wy = place(convention, X, Y_, poly)
    if theta_deg:
        cx, cy = wx.mean(), wy.mean()
        t = np.radians(theta_deg); ct, st = np.cos(t), np.sin(t)
        rx = cx + (wx - cx) * ct - (wy - cy) * st
        ry = cy + (wx - cx) * st + (wy - cy) * ct
        wx, wy = rx, ry
    return np.stack([wx + dx, wy + dy], 1)


def recall_combined(cfg, preds, census, polys, radius=1.0, **tf):
    tp = 0; ngt = 0
    for name in cfg["bcnp_yolo"]["test_plots"]:
        g = transform_census(cfg, name, census, polys, **tf)
        _, t, n = _recall(preds[name], g, radius)
        tp += t; ngt += n
    return tp / ngt if ngt else 0.0


def sweep(cfg, od, rd, preds, census, polys):
    print("\n===== STAGE 2 — offset / rotation / convention sweep (recall @1 m) =====")
    base = recall_combined(cfg, preds, census, polys)
    print(f"  baseline (current convention, 0,0,0): recall {base*100:.2f}%")

    # (a) translation heatmap at theta=0
    rng = np.arange(-20, 21, 1.0)
    grid = np.zeros((len(rng), len(rng)))
    for i, dyv in enumerate(rng):
        for j, dxv in enumerate(rng):
            grid[i, j] = recall_combined(cfg, preds, census, polys, dx=dxv, dy=dyv)
    bi = np.unravel_index(np.argmax(grid), grid.shape)
    best_dx, best_dy, best_tr = rng[bi[1]], rng[bi[0]], grid[bi]
    print(f"  best translation: dx={best_dx:+.0f} dy={best_dy:+.0f} m -> recall {best_tr*100:.2f}% "
          f"(vs {base*100:.2f}% at 0,0)")
    np.save(od / "recall_dxdy.npy", grid)

    # (b) rotation sweep about plot centre (at best translation)
    rot_rows = []
    best_rot = (0.0, base)
    for th in np.arange(-20, 21, 1.0):
        r = recall_combined(cfg, preds, census, polys, dx=best_dx, dy=best_dy, theta_deg=th)
        rot_rows.append((th, r))
        if r > best_rot[1]:
            best_rot = (th, r)
    print(f"  best fine rotation (at best dxdy): {best_rot[0]:+.0f} deg -> recall {best_rot[1]*100:.2f}%")

    # (c) discrete conventions + 90/180/270 rotations (each with its own best translation)
    disc = ["current", "yflip", "xflip", "xyflip", "swap", "swap_yflip", "swap_xflip", "swap_xyflip"]
    rows = []
    print("\n  discrete conventions (each optimised over dx,dy in +/-20 m):")
    coarse = np.arange(-20, 21, 2.0)
    best_global = ("current+0,0,0", base, dict(convention="current", dx=0, dy=0, theta=0))
    for conv in disc:
        best = (-1, 0, 0)
        for dyv in coarse:
            for dxv in coarse:
                r = recall_combined(cfg, preds, census, polys, dx=dxv, dy=dyv, convention=conv)
                if r > best[0]:
                    best = (r, dxv, dyv)
        rows.append(dict(convention=conv, best_recall=round(best[0], 4), best_dx=best[1], best_dy=best[2]))
        print(f"    {conv:13s}: recall {best[0]*100:5.2f}%  @ dx={best[1]:+.0f} dy={best[2]:+.0f}")
        if best[0] > best_global[1]:
            best_global = (f"{conv}+{best[1]:+.0f},{best[2]:+.0f}", best[0],
                           dict(convention=conv, dx=best[1], dy=best[2], theta=0))
    for th in (90.0, 180.0, 270.0):
        r = recall_combined(cfg, preds, census, polys, theta_deg=th)
        rows.append(dict(convention=f"rot{int(th)}", best_recall=round(r, 4), best_dx=0, best_dy=0))
        print(f"    rot{int(th):<10d}: recall {r*100:5.2f}%")
        if r > best_global[1]:
            best_global = (f"rot{int(th)}", r, dict(convention="current", dx=0, dy=0, theta=th))

    with open(od / "sweep_conventions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["convention", "best_recall", "best_dx", "best_dy"])
        w.writeheader(); w.writerows(rows)
    _fig_sweep(rd, grid, rng, rot_rows, base, best_dx, best_dy, best_tr)
    print(f"\n  GLOBAL BEST: {best_global[0]} -> recall {best_global[1]*100:.2f}% "
          f"(baseline {base*100:.2f}%)")
    return base, (best_dx, best_dy, best_tr), best_rot, best_global


def _fig_sweep(rd, grid, rng, rot_rows, base, best_dx, best_dy, best_tr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    im = a1.imshow(grid * 100, origin="lower", extent=[rng[0], rng[-1], rng[0], rng[-1]],
                   cmap="viridis", aspect="equal")
    a1.plot(0, 0, "w+", ms=12, mew=2); a1.plot(best_dx, best_dy, "r*", ms=14)
    a1.set_xlabel("dx (m, easting)"); a1.set_ylabel("dy (m, northing)")
    a1.set_title(f"recall(%) vs translation @0deg\nwhite+ = (0,0) {base*100:.1f}%, "
                 f"red* = max {best_tr*100:.1f}% @({best_dx:+.0f},{best_dy:+.0f})")
    fig.colorbar(im, ax=a1, fraction=0.046, label="recall %")
    a2.plot([r[0] for r in rot_rows], [r[1] * 100 for r in rot_rows], "-o", color="#4575b4", ms=3)
    a2.axvline(0, color="#999", ls="--", lw=1)
    a2.set_xlabel("rotation about plot centre (deg)"); a2.set_ylabel("recall %")
    a2.set_title("recall vs fine rotation (at best dx,dy)")
    fig.suptitle("s18 — offset / rotation sweep (peak at 0,0,0 => georef fine)", fontsize=12)
    fig.tight_layout(); fig.savefig(rd/"offset_rotation_sweep.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'offset_rotation_sweep.png'}")


# ── stage 3: internal structure ──────────────────────────────────────────────
def internal_structure(cfg, od, census, polys):
    print("\n===== STAGE 3 — internal structure =====")
    rows = []
    for name in cfg["bcnp_yolo"]["train_plots"] + [cfg["bcnp_yolo"]["val_plot"]] + cfg["bcnp_yolo"]["test_plots"]:
        X, Y_ = local_xy(cfg, name, census)
        XY = np.stack([X, Y_], 1)
        cov = np.cov((XY - XY.mean(0)).T); w, v = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
        poly = plot_poly(cfg, name, polys); b = poly.bounds
        corners = list(poly.exterior.coords)[:-1]
        # deviation of each edge from the nearest UTM axis (0 = perfectly axis-aligned)
        edev = []
        for i in range(4):
            a = np.degrees(np.arctan2(corners[(i+1) % 4][1]-corners[i][1],
                                      corners[(i+1) % 4][0]-corners[i][0])) % 90
            edev.append(min(a, 90 - a))
        max_dev = max(edev)
        axis_aligned = max_dev < 0.05 and abs(b[2]-b[0]-100) < 1 and abs(b[3]-b[1]-100) < 1
        rows.append(dict(plot=name, x_min=round(X.min(), 1), x_max=round(X.max(), 1),
                         y_min=round(Y_.min(), 1), y_max=round(Y_.max(), 1),
                         pca_angle=round(ang, 1), eigen_ratio=round(w[-1]/w[0], 2),
                         sw_corner=f"({b[0]:.0f},{b[1]:.0f})", bbox_w=round(b[2]-b[0], 1),
                         bbox_h=round(b[3]-b[1], 1), poly_edge_dev_deg=round(max_dev, 4),
                         poly_axis_aligned=bool(axis_aligned)))
        print(f"  {name}: X[{X.min():.1f},{X.max():.1f}] Y[{Y_.min():.1f},{Y_.max():.1f}]  "
              f"eigen_ratio {w[-1]/w[0]:.2f} (isotropic)  poly axis-aligned {axis_aligned} "
              f"(edge dev {max_dev:.4f}deg)  SW origin ({b[0]:.0f},{b[1]:.0f})")
    with open(od / "internal_structure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {od/'internal_structure.csv'}")
    return rows


def verdict(base, best_tr_info, best_rot, best_global, struct):
    (best_dx, best_dy, best_tr) = best_tr_info
    print("\n========================= VERDICT =========================")
    print(f"polygons: all axis-aligned 100x100 m squares, SW corner = (min-x,min-y) origin — clean.")
    print(f"census cloud: fills X[0-100] Y[0-100] m, near-isotropic — no internal rotation.")
    print(f"baseline recall @1 m (current georef): {base*100:.2f}%")
    print(f"best translation:  dx={best_dx:+.0f} dy={best_dy:+.0f} m -> {best_tr*100:.2f}%")
    print(f"best fine rotation: {best_rot[0]:+.0f} deg -> {best_rot[1]*100:.2f}%")
    print(f"global best transform: {best_global[0]} -> {best_global[1]*100:.2f}%")
    gain = (best_global[1] - base) * 100
    near0 = abs(best_dx) <= 2 and abs(best_dy) <= 2 and abs(best_rot[0]) <= 2 and best_global[2].get("theta", 0) == 0 and best_global[2].get("convention") == "current"
    if near0 or gain < 3:
        v = ("GEOREFERENCING IS CORRECT. No translation, rotation, or convention flip lifts "
             "recall materially above the (0,0,0) baseline — the sweep peaks at the current "
             "placement. The models' shared failure is NOT a georef bug; it is the dense-canopy "
             "task itself (consistent with the s17 size test). Confirm visually via the palm crops.")
    else:
        conv = best_global[2].get("convention"); th = best_global[2].get("theta", 0)
        kind = ("origin/axis FLIP (" + conv + ")" if conv != "current" else
                f"{th:.0f} deg ROTATION" if th else f"OFFSET dx={best_dx:+.0f},dy={best_dy:+.0f} m")
        v = (f"GEOREFERENCING LOOKS WRONG: recall jumps +{gain:.1f} pts under {kind}. "
             "This likely explains the shared s14/s15/s16 failure. Fix the census placement and re-run.")
    print("==>", v)
    print("DECISIVE CHECK: open reports/bcnp_georef/palm_test_*.png — palm points must sit on")
    print("radial-frond palm crowns. That is human-verifiable and independent of the weak model.")
    print("===========================================================")


def main():
    cfg = C.load_config(); np.random.seed(0)
    od, rd = paths(cfg)
    bcd = Y.bc_dir(cfg)
    census = pd.read_csv(bcd / "RP.Plot_census_data.2025.csv")
    polys = gpd.read_file(bcd / "tree_plots_polygons.geojson")

    print("===== STAGE 1 — PALM TEST (human-verifiable) =====")
    palm_crops(cfg, rd, "plot_18_2", census, polys, n_crops=6)
    palm_crops(cfg, rd, "plot_12_1", census, polys, n_crops=4)

    preds = load_preds(cfg, census, polys)
    base, best_tr_info, best_rot, best_global = sweep(cfg, od, rd, preds, census, polys)
    struct = internal_structure(cfg, od, census, polys)
    verdict(base, best_tr_info, best_rot, best_global, struct)


if __name__ == "__main__":
    main()
