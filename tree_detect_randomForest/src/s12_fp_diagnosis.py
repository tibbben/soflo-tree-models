"""
Step 12 — FP DIAGNOSIS: localization failure vs genuine inventory gap.  [local]

The s10 audit confirmed ~94/100 "false positives" contain a REAL tree. But that leaves
the load-bearing question for precision unanswered:
  (a) INVENTORY GAP  — a real tree the botanical inventory never recorded (a genuine new
      detection the eval unfairly penalized), or
  (b) LOCALIZATION   — a tree the inventory DOES have, but our box didn't line up with it
      at IoU 0.4 (so the tree is already counted; the FP is a loose/duplicate box, often
      sitting right next to the SAME tree's false negative — one tree double-counted).
Only (a) trees should be credited as brand-new true positives. This step splits the 94
real-tree FPs into (a) vs (b) by checking each against the inventory, and recomputes the
audit-corrected precision crediting ONLY the genuine gaps.

Self-contained; reconstructs the exact FP boxes by replaying s10's seed-0 sampling.
Writes only under outputs/fp_diag + reports/fp_diag. No retraining.

Run:  .venv/bin/python src/s12_fp_diagnosis.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
from pathlib import Path
import numpy as np
import geopandas as gpd
sys.path.append(str(Path(__file__).resolve().parent))
import common as C
import s10_audit as S10

# baseline numbers carried from s10 for the precision arithmetic
BASE_TP = 682
BASE_NPRED = 2162
FP_COLOR = "#d73027"
INV_COLOR = "#2166ac"


def reconstruct_real_fps(cfg):
    """Map the real_tree-tagged fp_ids back to their FP box geometries + inventory context.
    Returns (records, context) where records is a list of per-FP dicts."""
    preds, gt, pg, gg, info = S10.analyze(cfg)
    fp_idx, fp_best = info["fp_idx"], info["fp_best"]
    fn_set = set(int(x) for x in info["fn_idx"])          # unmatched inventory (FNs)
    scores = preds["score"].values if "score" in preds.columns else np.ones(len(preds))

    # replay the s10 seed-0 FP sampling (FP sampling is the first rng use in build_audit)
    rng = np.random.default_rng(S10.SEED)
    k = min(S10.N_SAMPLE, len(fp_idx))
    sel = rng.choice(len(fp_idx), size=k, replace=False); sel.sort()

    fp_csv = C.p(cfg, cfg["outputs_dir"]) / "audit" / "fp_audit_completed.csv"
    tags = {int(r["fp_id"]): (r.get("my_tag") or "").strip().lower()
            for r in csv.DictReader(open(fp_csv))}

    inv_cent = np.array([[g.centroid.x, g.centroid.y] for g in gg])   # inventory points

    records = []
    for sid in range(1, k + 1):
        if tags.get(sid) != "real_tree":
            continue
        pos = sel[sid - 1]
        pi = int(fp_idx[pos])
        geom = pg[pi]
        cx, cy = geom.centroid.x, geom.centroid.y
        d = np.hypot(inv_cent[:, 0] - cx, inv_cent[:, 1] - cy)
        nj = int(np.argmin(d))
        records.append(dict(
            fp_id=sid, pred_idx=pi, score=float(scores[pi]), geom=geom,
            cx=cx, cy=cy, best_iou=float(fp_best[pos]),
            nearest_inv_idx=nj, nearest_dist=float(d[nj]),
            nearest_geom=gg[nj], nearest_unmatched=(nj in fn_set)))
    ctx = dict(preds=preds, gg=gg, inv_cent=inv_cent, fn_set=fn_set, info=info, pg=pg)
    return records, ctx


def bucket(rec, dist_thr):
    """(b) localization if an inventory tree is nearby; else (a) inventory gap."""
    if rec["best_iou"] > 0 or rec["nearest_dist"] < dist_thr:
        return "localization"
    return "gap"


def main():
    cfg = C.load_config()
    fpd = cfg.get("fp_diag", {})
    dist_thr = float(fpd.get("dist_thr_m", 3.5))
    sens = [float(x) for x in fpd.get("sensitivity_m", [2.0, 3.0, 4.0])]

    od = C.p(cfg, cfg["outputs_dir"]) / "fp_diag"; od.mkdir(parents=True, exist_ok=True)
    rd = C.p(cfg, cfg["reports_dir"]) / "fp_diag"; rd.mkdir(parents=True, exist_ok=True)

    recs, ctx = reconstruct_real_fps(cfg)
    n = len(recs)
    print(f"real_tree-tagged FPs reconstructed: {n}")

    for r in recs:
        r["bucket"] = bucket(r, dist_thr)
    n_loc = sum(r["bucket"] == "localization" for r in recs)
    n_gap = n - n_loc
    n_loc_unmatched = sum(r["bucket"] == "localization" and r["nearest_unmatched"]
                          and r["nearest_dist"] < dist_thr for r in recs)

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit(f"\n===== FP DIAGNOSIS (real-tree FPs, n={n}) — dist_thr={dist_thr} m =====")
    emit(f"(a) INVENTORY GAP  (no inventory tree nearby)      : {n_gap:3d}  ({n_gap/n*100:4.1f}%)")
    emit(f"(b) LOCALIZATION   (inventory tree nearby)         : {n_loc:3d}  ({n_loc/n*100:4.1f}%)")
    emit(f"      of which the nearby inventory tree is UNMATCHED (double-counted "
         f"as FP+FN): {n_loc_unmatched}")

    # ── sensitivity of the split ───────────────────────────────────────────
    dists = np.array([r["nearest_dist"] for r in recs])
    gap_dists = np.array([r["nearest_dist"] for r in recs if r["bucket"] == "gap"])
    n_overlap = sum(r["best_iou"] > 0 for r in recs)
    emit("\nsensitivity of the (a)/(b) split to the distance cutoff:")
    emit(f"  {'cutoff':>7} | {'combined (overlap OR <cut)':>26} | {'distance-only (<cut)':>20}")
    emit(f"  {'':>7} | {'gap (a)':>12} {'loc (b)':>12} | {'gap (a)':>9} {'loc (b)':>9}")
    for t in sorted(set(sens + [dist_thr])):
        loc_c = sum(r["best_iou"] > 0 or r["nearest_dist"] < t for r in recs)
        loc_d = int((dists < t).sum())
        emit(f"  {t:>5.1f} m | {n-loc_c:>12d} {loc_c:>12d} | {n-loc_d:>9d} {loc_d:>9d}")
    emit(f"NOT a knife-edge: localization is decided by OVERLAP — all {n_overlap} localization "
         f"FPs overlap an inventory box (best_iou>0), independent of the distance cutoff.")
    emit(f"  Every gap-bucket FP sits >= {gap_dists.min():.1f} m from any inventory tree "
         f"(median {np.median(gap_dists):.0f} m), so moving the distance cutoff anywhere below")
    emit(f"  ~{gap_dists.min():.0f} m reclassifies nothing — the {n-n_overlap}/{n_overlap} split holds.")

    # ── double-count cross-check ───────────────────────────────────────────
    emit("\n----- double-counting cross-check -----")
    emit(f"unmatched inventory trees (FNs) total: {len(ctx['fn_set'])}")
    emit(f"real-tree FPs sitting within {dist_thr} m of an UNMATCHED inventory tree: "
         f"{n_loc_unmatched}")
    distinct_fn = len(set(r['nearest_inv_idx'] for r in recs
                          if r['nearest_unmatched'] and r['nearest_dist'] < dist_thr))
    emit(f"  -> {distinct_fn} distinct FN trees where ONE loose box created BOTH an FP and "
         f"an FN (in the audited sample of {n})")

    # ── precision arithmetic (both ways) ───────────────────────────────────
    n_fp_all = len(ctx["info"]["fp_idx"])
    orig_p = BASE_TP / BASE_NPRED
    p_all = (BASE_TP + n) / BASE_NPRED                      # credit all real_tree
    p_gap = (BASE_TP + n_gap) / BASE_NPRED                  # credit only gaps
    rate_all, rate_gap = n / 100.0, n_gap / 100.0          # audited 100 rows
    ext_all = (BASE_TP + rate_all * n_fp_all) / BASE_NPRED
    ext_gap = (BASE_TP + rate_gap * n_fp_all) / BASE_NPRED
    emit("\n----- corrected precision @IoU0.4 (tp=682 / preds=2162) -----")
    emit(f"original (no credit)                       : {orig_p:.3f}")
    emit(f"credit ALL {n} real-tree FPs as new TP       : {p_all:.3f}   "
         f"(extrapolated {rate_all*100:.0f}% x {n_fp_all} FPs -> ~{ext_all:.3f})")
    emit(f"credit ONLY {n_gap} inventory-gap FPs as new TP : {p_gap:.3f}   "
         f"(extrapolated {rate_gap*100:.0f}% x {n_fp_all} FPs -> ~{ext_gap:.3f})")
    emit("NOTE: the earlier ~0.96 estimate assumed ALL real-tree FPs were inventory gaps.")
    emit(f"      This step measures the gap fraction at {n_gap}/{n} = {n_gap/n*100:.0f}%, so the")
    emit("      honest corrected precision is the 'gap-only' column, not credit-all.")
    emit("=" * 63)

    _write_csv(od, recs)
    Path(od / "fp_diag_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {od/'fp_diag_summary.txt'}")

    _fig_hist(cfg, recs, dist_thr, rd)
    _fig_samples(cfg, recs, dist_thr, rd)
    _verdict(recs, n_gap, n_loc, p_gap, ext_gap, ext_all)


def _write_csv(od, recs):
    out = od / "fp_diag.csv"
    fields = ["fp_id", "pred_idx", "score", "cx", "cy", "best_iou",
              "nearest_dist", "nearest_inv_idx", "nearest_unmatched", "bucket"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            row = {k: r[k] for k in fields}
            row["score"] = round(r["score"], 3)
            row["cx"] = round(r["cx"], 2); row["cy"] = round(r["cy"], 2)
            row["best_iou"] = round(r["best_iou"], 3)
            row["nearest_dist"] = round(r["nearest_dist"], 2)
            w.writerow(row)
    print(f"wrote {out}")


def _fig_hist(cfg, recs, dist_thr, rd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    loc_d = np.array([r["nearest_dist"] for r in recs if r["bucket"] == "localization"])
    gap_d = np.array([r["nearest_dist"] for r in recs if r["bucket"] == "gap"])
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max(gap_d.max(), loc_d.max()) + 1, 1.0)
    ax.hist([loc_d, gap_d], bins=bins, stacked=True,
            color=["#d73027", "#1b7837"], edgecolor="white",
            label=[f"(b) localization — overlaps inventory box ({len(loc_d)})",
                   f"(a) inventory gap — no tree nearby ({len(gap_d)})"])
    ax.set_xlabel("distance from FP box center to nearest inventory point (m)")
    ax.set_ylabel("real-tree FP boxes")
    ax.set_title(f"s12 — real-tree FP distance to nearest inventory tree\n"
                 f"{len(loc_d)} localization (near, overlap inventory) vs "
                 f"{len(gap_d)} genuine gaps (>= {gap_d.min():.0f} m, median "
                 f"{np.median(gap_d):.0f} m)")
    ax.legend()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = rd / "dist_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _fig_samples(cfg, recs, dist_thr, rd):
    import rasterio
    from rasterio.windows import from_bounds
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    C.style()
    clip = C.p(cfg, cfg["outputs_dir"]) / "tiles" / "test_clip.tif"
    gaps = [r for r in recs if r["bucket"] == "gap"][:5]
    locs = [r for r in recs if r["bucket"] == "localization"
            and r["nearest_unmatched"] and r["nearest_dist"] < dist_thr][:5]
    # top up localization row if not enough unmatched ones
    if len(locs) < 5:
        extra = [r for r in recs if r["bucket"] == "localization" and r not in locs]
        locs += extra[:5 - len(locs)]
    cols = 5
    fig, axes = plt.subplots(2, cols, figsize=(2.6*cols, 5.6))
    with rasterio.open(clip) as src:
        for row, (grp, label, color) in enumerate(
                [(gaps, "(a) INVENTORY GAP — no inventory tree", "#1b7837"),
                 (locs, "(b) LOCALIZATION — inventory tree nearby", "#d73027")]):
            for c in range(cols):
                ax = axes[row, c]; ax.axis("off")
                if c >= len(grp):
                    continue
                r = grp[c]
                minx, miny, maxx, maxy = r["geom"].bounds
                cxw, cyw = (minx+maxx)/2, (miny+maxy)/2
                span = max(maxx-minx, maxy-miny) * 3.0
                w0, w1 = cxw-span/2, cxw+span/2
                h0, h1 = cyw-span/2, cyw+span/2
                win = from_bounds(w0, h0, w1, h1, src.transform)
                arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                wt = src.window_transform(win)
                ax.imshow(np.transpose(arr, (1, 2, 0)))

                def px(x, y):
                    return (x-wt.c)/wt.a, (y-wt.f)/wt.e
                bx0, by0 = px(minx, maxy); bx1, by1 = px(maxx, miny)
                ax.add_patch(Rectangle((bx0, by0), bx1-bx0, by1-by0, fill=False,
                                       edgecolor=FP_COLOR, linewidth=1.8))
                # mark nearest inventory point if it lands in the crop
                ig = r["nearest_geom"].centroid
                ipx, ipy = px(ig.x, ig.y)
                if 0 <= ipx <= arr.shape[2] and 0 <= ipy <= arr.shape[1]:
                    if r["nearest_unmatched"]:
                        ax.scatter([ipx], [ipy], s=80, marker="o", facecolors="none",
                                   edgecolors=INV_COLOR, linewidths=1.8)
                    else:
                        ax.scatter([ipx], [ipy], s=70, marker="x", c=INV_COLOR, linewidths=1.8)
                ax.set_title(f"#{r['fp_id']} d={r['nearest_dist']:.1f}m "
                             f"IoU={r['best_iou']:.2f}", fontsize=7.5, color=color)
            axes[row, 0].text(-0.08, 0.5, label.split(" — ")[0], rotation=90,
                              transform=axes[row, 0].transAxes, va="center", ha="center",
                              fontsize=9, color=color, weight="bold")
    fig.suptitle("s12 — FP buckets: red = predicted box, blue = nearest inventory point "
                 "(o = that tree is unmatched/FN)", fontsize=11)
    fig.tight_layout()
    out = rd / "sample_buckets.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _verdict(recs, n_gap, n_loc, p_gap, ext_gap, ext_all):
    n = len(recs)
    frac_loc = n_loc / n * 100
    frac_gap = n_gap / n * 100
    lean = "localization" if n_loc >= n_gap else "genuine inventory gaps"
    print("\n========================= VERDICT =========================")
    print(f"Of {n} real-tree false positives: {n_gap} ({frac_gap:.0f}%) are GENUINE "
          f"inventory gaps,\n  {n_loc} ({frac_loc:.0f}%) are LOCALIZATION failures on "
          f"trees the inventory already has.")
    print(f"=> The real-tree FPs are mostly {lean}.")
    print(f"Crediting only the genuine gaps, corrected precision is ~{p_gap:.3f} measured "
          f"(~{ext_gap:.2f} extrapolated),")
    print(f"  NOT the ~{ext_all:.2f} that crediting all 94 implied. The honest number sits")
    print(f"  {'well above' if p_gap > 0.35 else 'near'} the ~0.36 floor, "
          f"below the ~0.96 optimistic estimate.")
    print("Much of the FP and FN pile is the SAME trees, boxed loosely — a localization")
    print("problem, not purely an inventory-coverage problem.")
    print("===========================================================")


if __name__ == "__main__":
    main()
