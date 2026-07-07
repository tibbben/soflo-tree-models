"""
Step 10 — MANUAL AUDIT tool: is our eval understating accuracy?  [local]

Two questions the IoU-0.4 numbers can't answer on their own:
  * How many "false positives" are REAL trees the botanical inventory never recorded
    (inventory-coverage gap), not model errors?
  * How many "false negatives" are localization near-misses (loose boxes) vs trees the
    model genuinely never found?

This builds the evidence to answer them by eye, plus an auto-analysis that needs no human.
It uses the COMMITTED baseline predictions (outputs/predicted_trees.geojson, the original
non-swapped NE-test predict_tile run) matched to the 2211-box inventory at IoU 0.4.
Nothing is retrained; everything writes under outputs/audit + reports/audit.

Usage:
  build (default):  SOFLO_DATA_ROOT unneeded — crops come from outputs/tiles/test_clip.tif
      .venv/bin/python src/s10_audit.py
  re-score (after you fill outputs/audit/fp_audit.csv's my_tag column):
      .venv/bin/python src/s10_audit.py --rescore

Outputs:
  reports/audit/fp_grid_p*.png     numbered grids of sampled false positives
  reports/audit/fn_grid_p*.png     numbered grids of sampled false negatives
  outputs/audit/fp_audit.csv       [fp_id,best_iou,my_tag,notes]  my_tag: real_tree/error/unsure
  outputs/audit/fn_audit.csv       [fn_id,auto_bucket,my_tag,notes] my_tag: found_but_loose/
                                   truly_missed/not_a_tree/unsure
  outputs/audit/fn_buckets.csv     the auto FN bucket breakdown
  outputs/audit/{fp,fn}_crops/*.png  individual zoomable crops (gitignored)
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from shapely.strtree import STRtree
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

SEED = 0
IOU = 0.4
IOU_LOOSE = 0.30
PAD = 2.5          # crop spans PAD x the box size (context padding)
N_SAMPLE = 100
GRID_COLS, GRID_ROWS = 5, 5     # 25 per page


# ── matching ────────────────────────────────────────────────────────────────
def _iou(a, b):
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def greedy_match(pred_geoms, pred_scores, gt_geoms, thr):
    """Score-ordered greedy one-to-one IoU>=thr match.
    Returns (matched_pred:set, matched_gt:set, tp)."""
    order = np.argsort(-np.asarray(pred_scores))
    tree = STRtree(gt_geoms)
    used_gt, used_pred, tp = set(), set(), 0
    for i in order:
        p = pred_geoms[i]
        best, bj = thr, None
        for j in tree.query(p):
            if j in used_gt:
                continue
            v = _iou(p, gt_geoms[j])
            if v >= best:
                best, bj = v, j
        if bj is not None:
            used_gt.add(bj); used_pred.add(i); tp += 1
    return used_pred, used_gt, tp


def best_overlap(query_geoms, target_geoms):
    """For each query geom, the max IoU against any target geom (0 if none)."""
    tree = STRtree(target_geoms)
    out = np.zeros(len(query_geoms))
    for i, q in enumerate(query_geoms):
        best = 0.0
        for j in tree.query(q):
            v = _iou(q, target_geoms[j])
            if v > best:
                best = v
        out[i] = best
    return out


# ── data loading ────────────────────────────────────────────────────────────
def load(cfg):
    od = C.p(cfg, cfg["outputs_dir"])
    preds = gpd.read_file(od / "predicted_trees.geojson")
    gt = gpd.read_file(od / "boxes_test.geojson")
    pred_geoms = list(preds.geometry.values)
    pred_scores = preds["score"].values if "score" in preds.columns else np.ones(len(preds))
    gt_geoms = list(gt.geometry.values)
    return preds, gt, pred_geoms, pred_scores, gt_geoms


# ── 1) FN auto analysis ─────────────────────────────────────────────────────
def fn_bucket(best_iou):
    if best_iou >= IOU:
        return "contested (pred used by neighbor)"
    if best_iou >= IOU_LOOSE:
        return "near-miss / localization [0.30,0.40)"
    if best_iou > 0:
        return "poor localization (0,0.30)"
    return "true miss (not detected)"


def analyze(cfg):
    preds, gt, pg, ps, gg = load(cfg)
    n_pred, n_gt = len(pg), len(gg)
    matched_pred, matched_gt, tp = greedy_match(pg, ps, gg, IOU)
    _, _, tp03 = greedy_match(pg, ps, gg, IOU_LOOSE)

    fp_idx = np.array([i for i in range(n_pred) if i not in matched_pred])
    fn_idx = np.array([j for j in range(n_gt) if j not in matched_gt])

    # best-overlap both directions
    fn_best = best_overlap([gg[j] for j in fn_idx], pg) if len(fn_idx) else np.array([])
    fp_best = best_overlap([pg[i] for i in fp_idx], gg) if len(fp_idx) else np.array([])

    info = dict(n_pred=n_pred, n_gt=n_gt, tp=tp, tp03=tp03,
                fp_idx=fp_idx, fn_idx=fn_idx, fn_best=fn_best, fp_best=fp_best,
                recall04=tp / n_gt, recall03=tp03 / n_gt,
                precision04=tp / n_pred)
    return preds, gt, pg, gg, info


def print_fn_report(cfg, info):
    fn_best = info["fn_best"]
    buckets = ["true miss (not detected)", "poor localization (0,0.30)",
               "near-miss / localization [0.30,0.40)", "contested (pred used by neighbor)"]
    counts = {b: 0 for b in buckets}
    for v in fn_best:
        counts[fn_bucket(v)] += 1
    n_fn = len(fn_best)
    print("\n===== FN auto-analysis (why inventory trees are unmatched at IoU 0.4) =====")
    print(f"false negatives: {n_fn} / {info['n_gt']} inventory trees")
    rows = []
    for b in buckets:
        c = counts[b]
        pct = c / n_fn * 100 if n_fn else 0
        print(f"  {b:42s} {c:5d}  ({pct:5.1f}%)")
        rows.append({"bucket": b, "count": c, "pct": round(pct, 1)})
    print(f"\nrecall @ IoU 0.4 : {info['recall04']*100:5.1f}%  (tp {info['tp']})")
    print(f"recall @ IoU 0.3 : {info['recall03']*100:5.1f}%  (tp {info['tp03']})  "
          f"-> +{(info['recall03']-info['recall04'])*100:.1f} pts recovered by loosening IoU")
    print("Interpretation: the gap between recall@0.3 and recall@0.4 is trees the model")
    print("DID find but boxed loosely; the 'true miss' bucket is genuinely undetected.")
    # write bucket csv
    out = C.p(cfg, cfg["outputs_dir"]) / "audit" / "fn_buckets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "count", "pct"]); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    return counts


# ── crops + grids ───────────────────────────────────────────────────────────
def read_crop(src, box_geom, pad=PAD):
    minx, miny, maxx, maxy = box_geom.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    span = max(maxx - minx, maxy - miny) * pad
    w0, w1, h0, h1 = cx - span/2, cx + span/2, cy - span/2, cy + span/2
    win = from_bounds(w0, h0, w1, h1, src.transform)
    arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
    wt = src.window_transform(win)
    img = np.transpose(arr, (1, 2, 0)).astype(np.uint8)

    def to_px(x, y):
        return (x - wt.c) / wt.a, (y - wt.f) / wt.e
    bx0, by0 = to_px(minx, maxy)
    bx1, by1 = to_px(maxx, miny)
    ccx, ccy = to_px(cx, cy)
    return img, (bx0, by0, bx1, by1), (ccx, ccy)


def save_individual(img, box_px, path, color, mark_point=None):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = box_px
    d.rectangle([x0, y0, x1, y1], outline=color, width=2)
    if mark_point is not None:
        px, py = mark_point
        d.line([px-7, py, px+7, py], fill=color, width=2)
        d.line([px, py-7, px, py+7], fill=color, width=2)
    im.save(path)


def build_grid(items, clip_path, grid_dir, crop_dir, prefix, color, mark_center, title):
    """items: list of dicts with keys geom, id, sub (subtitle). Saves paginated grids
    + individual crops. Returns list of (id, sub) for the CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    C.style()
    crop_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    per_page = GRID_COLS * GRID_ROWS
    n = len(items)
    pages = (n + per_page - 1) // per_page
    rgb = {"red": (215, 48, 39), "lime": (150, 220, 20)}[color]

    with rasterio.open(clip_path) as src:
        crops = []
        for it in items:
            img, box_px, ctr = read_crop(src, it["geom"])
            crops.append((img, box_px, ctr))
            ipath = crop_dir / f"{prefix}_{it['id']:03d}.png"
            save_individual(img, box_px, ipath, rgb, ctr if mark_center else None)

    mpl_color = "#d73027" if color == "red" else "#7fdc14"
    for pg in range(pages):
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(GRID_COLS*2.4, GRID_ROWS*2.6))
        axes = np.atleast_2d(axes)
        for k in range(per_page):
            ax = axes[k // GRID_COLS, k % GRID_COLS]; ax.axis("off")
            idx = pg * per_page + k
            if idx >= n:
                continue
            img, box_px, ctr = crops[idx]
            it = items[idx]
            ax.imshow(img)
            x0, y0, x1, y1 = box_px
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                   edgecolor=mpl_color, linewidth=1.6))
            if mark_center:
                ax.plot(ctr[0], ctr[1], "+", color=mpl_color, ms=9, mew=1.6)
            ax.set_title(f"#{it['id']}  {it['sub']}", fontsize=7.5)
        fig.suptitle(f"{title}  (page {pg+1}/{pages})", fontsize=12)
        fig.tight_layout()
        out = grid_dir / f"{prefix}_grid_p{pg+1}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")


def build_audit(cfg):
    preds, gt, pg, gg, info = analyze(cfg)
    counts = print_fn_report(cfg, info)

    rng = np.random.default_rng(SEED)
    od = C.p(cfg, cfg["outputs_dir"]) / "audit"
    rd = C.p(cfg, cfg["reports_dir"]) / "audit"
    clip = C.p(cfg, cfg["outputs_dir"]) / "tiles" / "test_clip.tif"

    # ── FP sample ──────────────────────────────────────────────────────────
    fp_idx = info["fp_idx"]; fp_best = info["fp_best"]
    k_fp = min(N_SAMPLE, len(fp_idx))
    sel = rng.choice(len(fp_idx), size=k_fp, replace=False); sel.sort()
    fp_items, fp_rows = [], []
    for sid, s in enumerate(sel, 1):
        gi = int(fp_idx[s]); bi = float(fp_best[s])
        fp_items.append({"geom": pg[gi], "id": sid, "sub": f"IoU {bi:.2f}"})
        fp_rows.append({"fp_id": sid, "best_iou": round(bi, 3), "my_tag": "", "notes": ""})
    print(f"\nsampled {k_fp} of {len(fp_idx)} false positives (seed {SEED})")
    build_grid(fp_items, clip, rd, od / "fp_crops", "fp", "red", False,
               "s10 — FALSE POSITIVES (pred box; tag real_tree / error / unsure)")
    _write_template(od / "fp_audit.csv", ["fp_id", "best_iou", "my_tag", "notes"], fp_rows)

    # ── FN sample ──────────────────────────────────────────────────────────
    fn_idx = info["fn_idx"]; fn_best = info["fn_best"]
    k_fn = min(N_SAMPLE, len(fn_idx))
    sel = rng.choice(len(fn_idx), size=k_fn, replace=False); sel.sort()
    fn_items, fn_rows = [], []
    short = {"true miss (not detected)": "true-miss",
             "poor localization (0,0.30)": "poor-loc",
             "near-miss / localization [0.30,0.40)": "near-miss",
             "contested (pred used by neighbor)": "contested"}
    for sid, s in enumerate(sel, 1):
        gj = int(fn_idx[s]); bi = float(fn_best[s]); bucket = fn_bucket(bi)
        fn_items.append({"geom": gg[gj], "id": sid, "sub": f"{short[bucket]} {bi:.2f}"})
        fn_rows.append({"fn_id": sid, "auto_bucket": bucket, "my_tag": "", "notes": ""})
    print(f"sampled {k_fn} of {len(fn_idx)} false negatives (seed {SEED})")
    build_grid(fn_items, clip, rd, od / "fn_crops", "fn", "lime", True,
               "s10 — FALSE NEGATIVES (inventory point +; tag found_but_loose / truly_missed / not_a_tree / unsure)")
    _write_template(od / "fn_audit.csv", ["fn_id", "auto_bucket", "my_tag", "notes"], fn_rows)

    print("\nNEXT: open the grids in reports/audit/, then fill the my_tag column in")
    print("outputs/audit/fp_audit.csv and fn_audit.csv. Re-run with --rescore when done.")


def _write_template(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows, my_tag blank for you)")


# ── 4) re-score helper (idempotent) ─────────────────────────────────────────
def rescore(cfg):
    preds, gt, pg, gg, info = analyze(cfg)
    fp_csv = C.p(cfg, cfg["outputs_dir"]) / "audit" / "fp_audit.csv"
    if not fp_csv.exists():
        print(f"no {fp_csv} — run the build step first."); return
    tagged = {"real_tree": 0, "error": 0, "unsure": 0, "": 0, "other": 0}
    n = 0
    with open(fp_csv) as f:
        for r in csv.DictReader(f):
            n += 1
            t = (r.get("my_tag") or "").strip().lower()
            tagged[t if t in tagged else "other"] += 1
    n_real = tagged["real_tree"]
    n_labeled = n - tagged[""]
    tp, n_pred = info["tp"], info["n_pred"]
    orig_p = tp / n_pred
    corr_p = (tp + n_real) / n_pred        # credit audited real trees as TP
    print("\n===== RE-SCORE (inventory-gap corrected precision) =====")
    print(f"audited FP rows: {n}  (labeled {n_labeled}, blank {tagged['']})")
    print(f"  tagged real_tree={n_real}  error={tagged['error']}  unsure={tagged['unsure']}")
    print(f"original precision @IoU0.4 : {orig_p:.4f}  (tp {tp} / {n_pred})")
    print(f"corrected precision (audit): {corr_p:.4f}  (+{n_real} audited real trees as TP)")
    if n_labeled:
        rate = n_real / n_labeled
        est_real_all = rate * len(info["fp_idx"])
        est_p = (tp + est_real_all) / n_pred
        print(f"\nextrapolated (if the {rate*100:.0f}% real_tree rate holds across all "
              f"{len(info['fp_idx'])} FPs):")
        print(f"  est. real trees among all FPs ~ {est_real_all:.0f}  ->  "
              f"est. corrected precision ~ {est_p:.3f}")
    print("=========================================================")


def main():
    ap = argparse.ArgumentParser(description="treedetect manual audit tool")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute precision from filled fp_audit.csv (run after tagging)")
    args = ap.parse_args()
    cfg = C.load_config()
    if args.rescore:
        rescore(cfg)
    else:
        build_audit(cfg)


if __name__ == "__main__":
    main()
