"""
Step 11 — CLEAN evaluation region scaffold (exhaustive ground-truth bootstrap).  [local]

We proved the botanical inventory undercounts trees (~94% of "false positives" are real),
so inventory-based precision/recall is an unreliable ruler. This builds the tooling to
create an HONEST one: pick a small sub-area inside the NE test AOI, seed candidate boxes
from the committed baseline predictions (already ~94% real), and hand-verify EVERY tree in
QGIS so nothing is missed. Completeness over size — a small fully-labeled area beats a big
partial one. No retraining; the region's trees must stay OUT of any future training set.

Sub-commands (self-contained; writes only under outputs/clean_eval + reports/clean_eval):
  build (default)  pick the region, clip it from the ortho, export a QGIS-ready package
                   (predictions + inventory + an empty verified_trees layer) and numbered
                   render PNGs, and write a README with the labeling workflow.
  --coverage       given your edited verified_trees layer, report progress: trees verified,
                   predictions kept vs deleted, and brand-NEW trees (no pred + no inventory).
  --eval           score the baseline predictions against your clean GT (IoU 0.4 and 0.3)
                   next to the inventory-based number for the same region — the honest gap.

Run:  .venv/bin/python src/s11_clean_eval.py            # build
      .venv/bin/python src/s11_clean_eval.py --coverage # as you label
      .venv/bin/python src/s11_clean_eval.py --eval     # once labeled
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import argparse
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

PRED_COLOR = "#d73027"      # predicted boxes (red)
INV_COLOR = "#2166ac"       # inventory points (blue)
VER_COLOR = "#1b7837"       # verified (green)


# ── matching helpers ────────────────────────────────────────────────────────
def _iou(a, b):
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def greedy_pr(pred_geoms, pred_scores, gt_geoms, thr):
    """precision/recall/f1/tp for score-ordered greedy IoU>=thr matching."""
    n_pred, n_gt = len(pred_geoms), len(gt_geoms)
    if n_pred == 0 or n_gt == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, tp=0, n_pred=n_pred, n_gt=n_gt)
    from shapely.strtree import STRtree
    order = np.argsort(-np.asarray(pred_scores))
    tree = STRtree(gt_geoms)
    used, tp = set(), 0
    for i in order:
        p = pred_geoms[i]
        best, bj = thr, None
        for j in tree.query(p):
            if j in used:
                continue
            v = _iou(p, gt_geoms[j])
            if v >= best:
                best, bj = v, j
        if bj is not None:
            used.add(bj); tp += 1
    prec, rec = tp / n_pred, tp / n_gt
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, n_pred=n_pred, n_gt=n_gt)


def best_iou_to(query_geoms, target_geoms):
    if not len(target_geoms):
        return np.zeros(len(query_geoms))
    from shapely.strtree import STRtree
    tree = STRtree(target_geoms)
    out = np.zeros(len(query_geoms))
    for i, q in enumerate(query_geoms):
        b = 0.0
        for j in tree.query(q):
            v = _iou(q, target_geoms[j])
            if v > b:
                b = v
        out[i] = b
    return out


# ── paths / data ────────────────────────────────────────────────────────────
def paths(cfg):
    od = C.p(cfg, cfg["outputs_dir"]) / "clean_eval"
    rd = C.p(cfg, cfg["reports_dir"]) / "clean_eval"
    od.mkdir(parents=True, exist_ok=True); rd.mkdir(parents=True, exist_ok=True)
    return od, rd


def load_layers(cfg):
    root = C.p(cfg, cfg["outputs_dir"])
    preds = gpd.read_file(root / "predicted_trees.geojson")
    inv = gpd.read_file(root / "boxes_test.geojson")
    clip = root / "tiles" / "test_clip.tif"
    return preds, inv, clip


# ── region selection ────────────────────────────────────────────────────────
def pick_region(cfg, preds, clip):
    """Config bounds win; else auto-pick a size_m square with mixed land cover
    (heterogeneous prediction density: some dense canopy cells, some open)."""
    ce = cfg.get("clean_eval", {})
    size = float(ce.get("size_m", 250))
    if ce.get("bounds"):
        b = [float(v) for v in ce["bounds"]]
        print(f"region from config.clean_eval.bounds: {[round(v,1) for v in b]}")
        return b, size, "config"

    import rasterio
    cell = 50.0
    with rasterio.open(clip) as src:
        xmin, ymin, xmax, ymax = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
        nx = int(np.floor((xmax - xmin) / cell))
        ny = int(np.floor((ymax - ymin) / cell))
        # per-cell VALID (non-nodata) fraction so we never pick a region that runs off
        # the ortho's diagonal edge (those black pixels can't be labeled).
        sub_px = 5
        low = src.read([1, 2, 3], out_shape=(3, ny * sub_px, nx * sub_px))
    valid_px = (low.max(axis=0) > 8)              # any channel bright => real imagery
    valid = valid_px.reshape(ny, sub_px, nx, sub_px).mean(axis=(1, 3))
    valid = valid[::-1]                            # raster row 0 is top; grid row 0 is ymin

    cx = preds.geometry.centroid.x.values
    cy = preds.geometry.centroid.y.values
    counts = np.zeros((ny, nx))
    ix = np.clip(((cx - xmin) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((cy - ymin) / cell).astype(int), 0, ny - 1)
    for a, b in zip(iy, ix):
        counts[a, b] += 1

    win_cells = int(round(size / cell))          # 5 cells for 250 m
    dense_thr = np.percentile(counts[counts > 0], 75)
    best = None
    for r0 in range(0, ny - win_cells + 1):
        for c0 in range(0, nx - win_cells + 1):
            vsub = valid[r0:r0 + win_cells, c0:c0 + win_cells]
            if vsub.mean() < 0.98 or vsub.min() < 0.85:   # must be fully imaged
                continue
            sub = counts[r0:r0 + win_cells, c0:c0 + win_cells]
            total = sub.sum()
            if total < 10:
                continue
            cv = sub.std() / (sub.mean() + 1e-6)  # heterogeneity: mix of dense + open
            n_open = int((sub <= 1).sum())
            n_dense = int((sub >= dense_thr).sum())
            mix = min(n_open, n_dense)            # want BOTH open and dense cells
            score = (mix, round(cv, 3), total)
            if best is None or score > best[0]:
                bx = xmin + c0 * cell
                by = ymin + r0 * cell
                best = (score, [bx, by, bx + size, by + size])
    if best is None:
        raise RuntimeError("no fully-imaged candidate region found; set clean_eval.bounds")
    b = best[1]
    print(f"auto-picked region (mixed land cover, fully imaged): {[round(v,1) for v in b]}")
    print(f"  score (open_cells, cv, total_preds) = "
          f"({best[0][0]}, {best[0][1]:.2f}, {int(best[0][2])})")
    return b, size, "auto"


def clip_region(a, region):
    """subset a GeoDataFrame to boxes whose centroid falls in the region."""
    minx, miny, maxx, maxy = region
    c = a.geometry.centroid
    m = (c.x >= minx) & (c.x <= maxx) & (c.y >= miny) & (c.y <= maxy)
    return a[m].copy().reset_index(drop=True)


# ── build ───────────────────────────────────────────────────────────────────
def build(cfg):
    od, rd = paths(cfg)
    preds, inv, clip = load_layers(cfg)
    region, size, src_of = pick_region(cfg, preds, clip)
    minx, miny, maxx, maxy = region

    rp = clip_region(preds, region)
    ri = clip_region(inv, region)
    print(f"\nregion {size:.0f}x{size:.0f} m  bounds {[round(v,1) for v in region]}")
    print(f"  seed predictions in region : {len(rp)}")
    print(f"  inventory trees in region  : {len(ri)}")

    # region definition (committed)
    reg_gdf = gpd.GeoDataFrame(
        {"name": ["clean_eval_region"], "size_m": [size], "source": [src_of]},
        geometry=[box(*region)], crs=cfg["crs"])
    reg_gdf.to_file(od / "region.geojson", driver="GeoJSON")

    # seed predictions layer (editable): pred_id + score + keep flag
    rp = rp.reset_index(drop=True)
    rp_out = gpd.GeoDataFrame(
        {"pred_id": range(1, len(rp) + 1),
         "score": rp["score"].round(3) if "score" in rp.columns else 1.0,
         "keep": ["" for _ in range(len(rp))]},
        geometry=rp.geometry.values, crs=cfg["crs"])
    rp_out.to_file(od / "region_predictions.geojson", driver="GeoJSON")

    # inventory layer — points (centroids) so it reads as an overlay in QGIS
    ipts = ri.geometry.centroid
    ri_out = gpd.GeoDataFrame(
        {"inv_id": range(1, len(ri) + 1),
         "species": ri["species"] if "species" in ri.columns else ""},
        geometry=list(ipts.values), crs=cfg["crs"])
    ri_out.to_file(od / "region_inventory.geojson", driver="GeoJSON")
    # keep inventory BOXES too (for IoU eval later), not just points
    gpd.GeoDataFrame({"inv_id": range(1, len(ri) + 1)},
                     geometry=ri.geometry.values, crs=cfg["crs"]).to_file(
        od / "region_inventory_boxes.geojson", driver="GeoJSON")

    # empty verified_trees layer (you draw/edit these in QGIS) — polygon, right CRS
    ver = gpd.GeoDataFrame({"tree_id": [], "status": [], "notes": []},
                           geometry=[], crs=cfg["crs"])
    ver.to_file(od / "verified_trees.gpkg", driver="GPKG", layer="verified_trees")

    # clipped raster (GITIGNORED — large)
    _clip_raster(clip, region, od / "region_clip.tif")

    _render(cfg, od / "region_clip.tif", rp_out, ri_out, region, rd)
    _write_readme(cfg, od, region, size, len(rp), len(ri))
    print("\nNEXT: open outputs/clean_eval/ in QGIS (see README.md) and label. Then run")
    print("  --coverage  to track progress, and  --eval  once every tree is verified.")


def _clip_raster(clip, region, out):
    import rasterio
    from rasterio.windows import from_bounds
    minx, miny, maxx, maxy = region
    with rasterio.open(clip) as src:
        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(window=win)
        prof = src.profile.copy()
        prof.update(height=data.shape[1], width=data.shape[2],
                    transform=src.window_transform(win))
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(data)
    print(f"wrote {out}  ({data.shape[2]}x{data.shape[1]} px, gitignored)")


def _render(cfg, clip_tif, rp, ri, region, rd):
    """Numbered overview + 2x2 quadrant PNGs: predicted boxes + inventory points."""
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    C.style()
    minx, miny, maxx, maxy = region

    with rasterio.open(clip_tif) as src:
        W, H = src.width, src.height
        step = max(1, int(round(max(W, H) / 2400)))
        img = src.read([1, 2, 3], out_shape=(3, H // step, W // step))
        wt = src.transform

    def to_px(x, y):
        col = (x - wt.c) / wt.a / step
        row = (y - wt.f) / wt.e / step
        return col, row

    def draw(ax, xlim=None, ylim=None, number=True):
        ax.imshow(np.transpose(img, (1, 2, 0)))
        for i, g in enumerate(rp.geometry.values, 1):
            x0, y0 = to_px(g.bounds[0], g.bounds[3])
            x1, y1 = to_px(g.bounds[2], g.bounds[1])
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                   edgecolor=PRED_COLOR, linewidth=0.8))
            if number:
                ax.text(x0, y0-1, str(i), color=PRED_COLOR, fontsize=5, va="bottom")
        px = [to_px(p.x, p.y) for p in ri.geometry.values]
        if px:
            ax.scatter([p[0] for p in px], [p[1] for p in px], s=14, marker="+",
                       c=INV_COLOR, linewidths=0.8)
        ax.axis("off")
        if xlim:
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    legend = [Line2D([0], [0], color=PRED_COLOR, lw=2, label=f"predicted boxes ({len(rp)})"),
              Line2D([0], [0], marker="+", ls="", c=INV_COLOR, label=f"inventory points ({len(ri)})")]

    fig, ax = plt.subplots(figsize=(12, 12))
    draw(ax)
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    ax.set_title(f"s11 — clean-eval region {region[2]-region[0]:.0f}x{region[3]-region[1]:.0f} m\n"
                 "seed predictions (red) + inventory points (blue) — verify EVERY tree in QGIS")
    fig.tight_layout()
    out = rd / "region_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    # 2x2 quadrants for legible numbering
    hpx, wpx = img.shape[1], img.shape[2]
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    quads = [((0, wpx/2), (hpx/2, 0)), ((wpx/2, wpx), (hpx/2, 0)),
             ((0, wpx/2), (hpx, hpx/2)), ((wpx/2, wpx), (hpx, hpx/2))]
    for ax, (xl, yl) in zip(axes.ravel(), quads):
        draw(ax, xl, yl)
    fig.suptitle("s11 — clean-eval region quadrants (zoom for numbered predicted boxes)", fontsize=13)
    fig.tight_layout()
    out = rd / "region_quadrants.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _write_readme(cfg, od, region, size, n_pred, n_inv):
    txt = f"""# Clean evaluation region — labeling workflow

Goal: label EVERY tree in this {size:.0f}x{size:.0f} m square so we have an HONEST test set.
The botanical inventory undercounts trees (we measured ~94% of "false positives" are real),
so we cannot trust inventory-based precision/recall. Completeness matters more than size —
label this small area with ZERO gaps.

Region bounds (EPSG:32617): {[round(v,1) for v in region]}
Seeded from {n_pred} baseline predictions + {n_inv} inventory trees.

## Files in this folder
- region_clip.tif            the imagery for this region (GITIGNORED — regenerate with build)
- region.geojson             the region boundary
- region_predictions.geojson candidate boxes from the model (EDIT THESE — ~94% are real)
- region_inventory.geojson   inventory tree points (reference; blue)
- region_inventory_boxes.geojson  inventory boxes (used by --eval only)
- verified_trees.gpkg        EMPTY layer for your final ground truth (draw/keep boxes here)
- ../..​/reports/clean_eval/region_overview.png + region_quadrants.png  printed references

## Open in QGIS
1. Layer > Add Raster Layer…  -> region_clip.tif   (your basemap)
2. Layer > Add Vector Layer…  -> region_predictions.geojson  (red candidate boxes)
3. Layer > Add Vector Layer…  -> region_inventory.geojson    (blue reference points)
4. Layer > Add Vector Layer…  -> verified_trees.gpkg         (your editable GT layer)
   All layers are already EPSG:32617; QGIS should align them on the raster.

## Workflow — build verified_trees so EVERY tree has exactly one box
Toggle editing (pencil) on verified_trees, then for every tree in the imagery:
  - ACCEPT a good prediction: copy the prediction box into verified_trees
    (or draw a box over that crown). status = "kept".
  - REJECT a false box: simply do not add it (leave it only in predictions). We will
    detect deletions automatically by comparing verified_trees to region_predictions.
  - ADD a miss: if a tree has NO prediction box, DRAW a new box. status = "added".
    These are the inventory's misses — the trees we most need captured.
  - status field values (free text is fine): kept / added / adjusted
Save the layer (Ctrl+S) often. There is no wrong pace — re-run the helpers anytime.

## Track progress
    .venv/bin/python src/s11_clean_eval.py --coverage
reports: trees verified, predictions kept vs deleted, brand-NEW trees (no pred + no inventory).

## Get the honest score (once every tree is verified)
    .venv/bin/python src/s11_clean_eval.py --eval
prints precision/recall/F1 of the baseline predictions against your clean GT at IoU 0.4 and
0.3, next to the inventory-based number for the same region — the gap is the eval error.

IMPORTANT: keep these trees OUT of any future training set — this is the held-out honest test.
"""
    (od / "README.md").write_text(txt)
    print(f"wrote {od / 'README.md'}")


# ── verified layer loading (gpkg or geojson) ────────────────────────────────
def _load_verified(od):
    for name in ("verified_trees.gpkg", "verified_trees.geojson"):
        p = od / name
        if p.exists():
            try:
                g = gpd.read_file(p)
                return g, name
            except Exception:
                continue
    return None, None


# ── coverage ────────────────────────────────────────────────────────────────
def coverage(cfg):
    od, _ = paths(cfg)
    ver, src = _load_verified(od)
    if ver is None:
        print("no verified_trees layer yet — run build first."); return
    preds = gpd.read_file(od / "region_predictions.geojson")
    invb = gpd.read_file(od / "region_inventory_boxes.geojson")
    n_ver = len(ver)
    print(f"\n===== CLEAN-EVAL COVERAGE ({src}) =====")
    print(f"verified trees drawn        : {n_ver}")
    if n_ver == 0:
        print("  (start labeling in QGIS — see outputs/clean_eval/README.md)")
        print(f"  seed predictions available: {len(preds)}   inventory points: {len(invb)}")
        print("=========================================")
        return
    vg = list(ver.geometry.values)
    pg = list(preds.geometry.values)
    ig = list(invb.geometry.values)

    # predictions kept (a verified box overlaps them) vs deleted (none)
    pred_best = best_iou_to(pg, vg)
    kept = int((pred_best >= 0.3).sum())
    deleted = len(pg) - kept
    # verified provenance: overlaps a prediction? an inventory tree?
    v_pred = best_iou_to(vg, pg)
    v_inv = best_iou_to(vg, ig)
    from_pred = int((v_pred >= 0.3).sum())
    new_trees = int(((v_pred < 0.3) & (v_inv < 0.3)).sum())
    from_inv_only = int(((v_pred < 0.3) & (v_inv >= 0.3)).sum())
    print(f"predictions kept (matched)  : {kept} / {len(pg)}")
    print(f"predictions deleted (false) : {deleted} / {len(pg)}")
    print(f"verified matching a pred     : {from_pred}")
    print(f"verified = inventory miss    : {from_inv_only}  (had inventory pt, no pred)")
    print(f"verified = brand NEW tree    : {new_trees}  (no pred AND no inventory)")
    print("=========================================")


# ── eval ────────────────────────────────────────────────────────────────────
def clean_scores(cfg):
    od, _ = paths(cfg)
    ver, src = _load_verified(od)
    preds = gpd.read_file(od / "region_predictions.geojson")
    invb = gpd.read_file(od / "region_inventory_boxes.geojson")
    root = C.p(cfg, cfg["outputs_dir"])
    allpred = gpd.read_file(root / "predicted_trees.geojson")
    # scores for the region preds (match by pred order preserved in region_predictions)
    pg = list(preds.geometry.values)
    ps = preds["score"].values if "score" in preds.columns else np.ones(len(preds))
    iou_list = cfg.get("clean_eval", {}).get("iou_eval", [0.4, 0.3])

    if ver is None or len(ver) == 0:
        print("no verified trees yet — label in QGIS first, then re-run --eval.")
        print("(coverage helper: --coverage)")
        return
    vg = list(ver.geometry.values)
    ig = list(invb.geometry.values)
    delta = len(vg) - len(ig)
    gap = (f"{delta} more than inventory — trees it missed" if delta > 0 else
           f"{-delta} fewer than inventory (partial labeling?)" if delta < 0 else
           "same count as inventory")
    print(f"\n===== HONEST EVAL — baseline predictions in the clean region =====")
    print(f"clean GT (verified): {len(vg)} trees   |   inventory GT: {len(ig)} trees "
          f"({gap})")
    print(f"predictions scored : {len(pg)}\n")
    hdr = f"{'IoU':>5} | {'clean P':>8} {'clean R':>8} {'clean F1':>9} | " \
          f"{'inv P':>7} {'inv R':>7} {'inv F1':>7}"
    print(hdr); print("-" * len(hdr))
    for iou in iou_list:
        cs = greedy_pr(pg, ps, vg, iou)
        isc = greedy_pr(pg, ps, ig, iou)
        print(f"{iou:>5} | {cs['precision']:>8.3f} {cs['recall']:>8.3f} {cs['f1']:>9.3f} | "
              f"{isc['precision']:>7.3f} {isc['recall']:>7.3f} {isc['f1']:>7.3f}")
    print("\nclean = vs your exhaustively-verified GT;  inv = vs the (incomplete) inventory.")
    print("The precision gap is real trees the inventory omitted; the recall gap is boxing.")


def main():
    ap = argparse.ArgumentParser(description="treedetect clean-eval region scaffold")
    ap.add_argument("--coverage", action="store_true", help="report labeling progress")
    ap.add_argument("--eval", action="store_true", help="honest precision/recall vs verified GT")
    args = ap.parse_args()
    cfg = C.load_config()
    if args.coverage:
        coverage(cfg)
    elif args.eval:
        clean_scores(cfg)
    else:
        build(cfg)


if __name__ == "__main__":
    main()
