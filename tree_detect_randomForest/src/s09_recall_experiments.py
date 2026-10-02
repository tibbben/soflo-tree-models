"""
Step 09 — NO-RETRAIN recall experiments on the held-out NE test AOI.  [local]

Three inference-time levers to raise detection recall WITHOUT retraining, each
varied one at a time and all evaluated identically (IoU 0.4 vs the inventory):
  A. score/confidence threshold sweep
  B. predict-time tiling (patch_overlap, and one alternate patch_size)
  C. resolution matching — resample the 5 cm test ortho toward the ~10 cm the
     released DeepForest weights expect, predict, map boxes back via the resampled
     raster transform, evaluate against the SAME world-coord inventory.
Plus an optional combined run (best-F1 setting from each) to see if gains stack.

Fixed: original (non-swapped) best checkpoint; original split (SW train / NE test);
evaluate on NE; IoU 0.4. Everything writes to outputs/recall_exp + reports/recall_exp,
never the committed baseline.

Two baselines are reported:
  * evaluate  — DeepForest's own per-tile evaluate(); reproduces the committed
    0.400 / 0.386 and proves the checkpoint/data did not drift (integrity gate).
  * predict_tile — whole-AOI windowed prediction scored against the UNIQUE inventory
    boxes; reads lower than the per-tile number (stricter denominator: it covers the
    whole AOI, not just curated non-empty tiles). This is the anchor the A/B/C
    experiments are compared against, because every lever here is a whole-AOI
    inference-time knob.

Run:  SOFLO_DATA_ROOT=/path .venv/bin/python src/s09_recall_experiments.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
import time
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from affine import Affine
from shapely.strtree import STRtree
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

IOU = 0.4
DEF_THRESH = 0.30
DEF_OVERLAP = 0.10
PATCH_M = 40.0                 # ground footprint of a predict window (kept constant)
BASELINE_PR = (0.3997, 0.3856)  # committed eval_summary.csv
TOL = 0.03                     # integrity tolerance for the evaluate() reproduction


# ── evaluation: greedy one-to-one IoU matcher in world coords ───────────────
def _iou(a, b):
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def match_pr(pred_geoms, pred_scores, gt_geoms, iou=IOU):
    """precision / recall / f1 / tp via score-ordered greedy IoU>=iou matching."""
    n_pred, n_gt = len(pred_geoms), len(gt_geoms)
    if n_pred == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, tp=0, predicted=0)
    order = np.argsort(-np.asarray(pred_scores))
    tree = STRtree(gt_geoms)
    used = set(); tp = 0
    for i in order:
        p = pred_geoms[i]
        best, bj = iou, None
        for j in tree.query(p):
            if j in used:
                continue
            v = _iou(p, gt_geoms[j])
            if v >= best:
                best, bj = v, j
        if bj is not None:
            used.add(bj); tp += 1
    prec = tp / n_pred
    rec = tp / n_gt
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, predicted=n_pred)


# ── prediction on an arbitrary raster (clip or resampled clip) ──────────────
def predict_raster(model, raster_path, score_thresh, patch_overlap, patch_m=PATCH_M):
    # predict_tile reads the torchvision RetinaNet attribute m.model.score_thresh,
    # NOT m.config["score_thresh"] — set both so the threshold actually takes effect.
    model.model.score_thresh = score_thresh
    model.config["score_thresh"] = score_thresh
    with rasterio.open(raster_path) as src:
        transform = src.transform
        gsd = abs(src.transform.a)
    patch_px = int(round(patch_m / gsd))
    pred = model.predict_tile(path=str(raster_path), patch_size=patch_px,
                              patch_overlap=patch_overlap)
    if pred is None or len(pred) == 0:
        return [], [], patch_px
    pred = pred.reset_index(drop=True)
    geoms = [C.pixel_box_to_world(r.xmin, r.ymin, r.xmax, r.ymax, transform)
             for _, r in pred.iterrows()]
    scores = pred["score"].values if "score" in pred.columns else np.ones(len(pred))
    return geoms, scores, patch_px


def resample_clip(src_path, dst_path, target_gsd):
    """Resample a clip to target_gsd (m/px), preserving CRS/georeferencing."""
    with rasterio.open(src_path) as src:
        gsd = abs(src.transform.a)
        scale = gsd / target_gsd                       # <1 => downsample
        new_w = max(1, int(round(src.width * scale)))
        new_h = max(1, int(round(src.height * scale)))
        data = src.read(out_shape=(src.count, new_h, new_w),
                        resampling=Resampling.bilinear)
        new_transform = src.transform * Affine.scale(src.width / new_w, src.height / new_h)
        prof = src.profile.copy()
        prof.update(width=new_w, height=new_h, transform=new_transform)
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(data)
    return dst_path


def main():
    cfg = C.load_config()
    from deepforest import main as df_main

    od = C.p(cfg, cfg["outputs_dir"]); rd = C.p(cfg, cfg["reports_dir"])
    exp_out = od / "recall_exp"; exp_out.mkdir(parents=True, exist_ok=True)
    exp_rep = rd / "recall_exp"; exp_rep.mkdir(parents=True, exist_ok=True)

    clip = od / "tiles" / "test_clip.tif"
    test_dir = od / "tiles" / "test"
    test_csv = test_dir / "test_tiles.csv"

    ckpt = C.best_checkpoint(od / "model")
    print(f"checkpoint: {ckpt.name}")
    m = df_main.deepforest.load_from_checkpoint(str(ckpt))

    gt = gpd.read_file(od / "boxes_test.geojson")
    with rasterio.open(clip) as src:
        assert src.crs == gt.crs or str(src.crs) == str(gt.crs), "CRS mismatch clip vs GT"
    gt_geoms = list(gt.geometry.values)
    n_gt = len(gt_geoms)
    print(f"NE inventory GT boxes: {n_gt}")

    rows = []

    def record(config, geoms, scores, threshold, overlap, patch_m, res_cm, method):
        r = match_pr(geoms, scores, gt_geoms)
        row = dict(config=config, method=method, threshold=threshold,
                   patch_overlap=overlap, patch_size_m=patch_m, resolution_cm=res_cm,
                   precision=round(r["precision"], 4), recall=round(r["recall"], 4),
                   f1=round(r["f1"], 4), predicted_count=r["predicted"], tp=r["tp"])
        rows.append(row)
        print(f"  {config:28s} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} pred={r['predicted']}")
        return row

    # ── 0) baseline integrity via DeepForest evaluate() ────────────────────
    print("\n[0] baseline — DeepForest per-tile evaluate() (integrity gate)")
    m.config["score_thresh"] = DEF_THRESH
    ev = m.evaluate(str(test_csv), iou_threshold=IOU, root_dir=str(test_dir))
    evP, evR = float(ev["box_precision"]), float(ev["box_recall"])
    print(f"  evaluate P={evP:.4f} R={evR:.4f}  (committed {BASELINE_PR[0]}/{BASELINE_PR[1]})")
    drift = abs(evP - BASELINE_PR[0]) > TOL or abs(evR - BASELINE_PR[1]) > TOL
    rows.append(dict(config="baseline_evaluate_pertile", method="evaluate",
                     threshold=DEF_THRESH, patch_overlap="(per-tile)", patch_size_m=PATCH_M,
                     resolution_cm=5.0, precision=round(evP, 4), recall=round(evR, 4),
                     f1=round(2*evP*evR/(evP+evR), 4), predicted_count="", tp=""))
    if drift:
        print("\n!!! STOP: evaluate() did NOT reproduce the committed baseline within "
              f"{TOL}. Something changed (checkpoint/data/env). Aborting experiments.")
        _write_csv(exp_out, rows)
        return
    print("  integrity OK — committed baseline reproduced.")

    # predict_tile baseline anchor (whole-AOI; the A/B/C reference)
    print("\n[0b] baseline — whole-AOI predict_tile (experiment anchor)")
    g, s, _ = predict_raster(m, clip, DEF_THRESH, DEF_OVERLAP)
    base = record("baseline_predict_tile", g, s, DEF_THRESH, DEF_OVERLAP, PATCH_M, 5.0, "predict_tile")
    base_recall = base["recall"]

    # ── A) score threshold sweep ───────────────────────────────────────────
    print("\n[A] score threshold sweep (default patch 40m / overlap 0.10)")
    for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
        if t == DEF_THRESH:
            r = dict(base); r["config"] = f"A_thresh_{t:.2f}"; rows.append(r)
            print(f"  {r['config']:28s} P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"F1={r['f1']:.3f} pred={r['predicted_count']}  (= baseline)")
            continue
        g, s, _ = predict_raster(m, clip, t, DEF_OVERLAP)
        record(f"A_thresh_{t:.2f}", g, s, t, DEF_OVERLAP, PATCH_M, 5.0, "predict_tile")

    # ── B) patch overlap + alternate patch size ────────────────────────────
    print("\n[B] tiling sweep (default threshold 0.30)")
    for ov in [0.10, 0.25, 0.40]:
        if ov == DEF_OVERLAP:
            r = dict(base); r["config"] = f"B_overlap_{ov:.2f}"; rows.append(r)
            print(f"  {r['config']:28s} P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"F1={r['f1']:.3f} pred={r['predicted_count']}  (= baseline)")
            continue
        g, s, _ = predict_raster(m, clip, DEF_THRESH, ov)
        record(f"B_overlap_{ov:.2f}", g, s, DEF_THRESH, ov, PATCH_M, 5.0, "predict_tile")
    # alternate patch size (smaller window -> more, tighter tiles)
    alt_m = 30.0
    g, s, _ = predict_raster(m, clip, DEF_THRESH, DEF_OVERLAP, patch_m=alt_m)
    record(f"B_patch_{alt_m:.0f}m", g, s, DEF_THRESH, DEF_OVERLAP, alt_m, 5.0, "predict_tile")

    # ── C) resolution matching ─────────────────────────────────────────────
    print("\n[C] resolution matching (resample test clip, default threshold/overlap)")
    res_results = {}
    for cm in (7.5, 10.0):
        dst = exp_out / f"test_clip_{cm:g}cm.tif"
        print(f"  resampling -> {cm} cm/px ({dst.name}) ...", flush=True)
        resample_clip(clip, dst, cm / 100.0)
        g, s, ppx = predict_raster(m, dst, DEF_THRESH, DEF_OVERLAP)
        row = record(f"C_resolution_{cm:g}cm", g, s, DEF_THRESH, DEF_OVERLAP, PATCH_M, cm, "predict_tile")
        res_results[cm] = row

    # ── combined: best-F1 setting from each of A/B/C ───────────────────────
    print("\n[combined] best-F1 of A (threshold) + B (tiling) + C (resolution)")
    def best(prefix):
        cand = [r for r in rows if r["config"].startswith(prefix)]
        return max(cand, key=lambda r: r["f1"])
    bA, bB, bC = best("A_"), best("B_"), best("C_")
    bt = bA["threshold"]
    bov = bB["patch_overlap"] if isinstance(bB["patch_overlap"], float) else DEF_OVERLAP
    bpm = bB["patch_size_m"]
    bcm = bC["resolution_cm"]
    print(f"  picked: threshold={bt}  overlap={bov}  patch={bpm}m  resolution={bcm}cm")
    if bcm != 5.0:
        comb_raster = exp_out / f"test_clip_{bcm:g}cm.tif"
    else:
        comb_raster = clip
    g, s, _ = predict_raster(m, comb_raster, bt, bov, patch_m=bpm)
    record("combined_best", g, s, bt, bov, bpm, bcm, "predict_tile")

    _write_csv(exp_out, rows)
    _fig_threshold(cfg, rows)
    _fig_by_config(cfg, rows, base_recall, evR)
    _summary(rows, base, evP, evR)


def _write_csv(exp_out, rows):
    out = exp_out / "recall_comparison.csv"
    fields = ["config", "method", "threshold", "patch_overlap", "patch_size_m",
              "resolution_cm", "precision", "recall", "f1", "predicted_count", "tp"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nwrote {out}")


def _fig_threshold(cfg, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    A = sorted([r for r in rows if r["config"].startswith("A_thresh")],
               key=lambda r: r["threshold"])
    if not A:
        return
    th = [r["threshold"] for r in A]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(th, [r["recall"] for r in A], "-o", color="#d73027", label="recall")
    ax1.plot(th, [r["precision"] for r in A], "-o", color="#4575b4", label="precision")
    ax1.plot(th, [r["f1"] for r in A], "-o", color="#7b3294", label="F1")
    ax1.set_xlabel("score threshold"); ax1.set_ylabel("score (IoU 0.4)")
    ax1.invert_xaxis()  # lower threshold (more boxes) to the right
    ax1.legend(loc="center left")
    ax2 = ax1.twinx()
    ax2.plot(th, [r["predicted_count"] for r in A], "--s", color="#999999", ms=3,
             label="predicted count")
    ax2.set_ylabel("predicted boxes")
    ax2.legend(loc="center right")
    ax1.set_title("s09 — Exp A: precision / recall / F1 vs score threshold")
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "recall_exp" / "recall_vs_threshold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _fig_by_config(cfg, rows, base_recall, ev_recall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    show = [r for r in rows if r["method"] == "predict_tile"]
    labels = [r["config"].replace("predict_tile", "pt") for r in show]
    rec = [r["recall"] for r in show]
    colors = []
    for r in show:
        c = "#999999"
        if r["config"].startswith("A_"): c = "#d73027"
        elif r["config"].startswith("B_"): c = "#f1a340"
        elif r["config"].startswith("C_"): c = "#1b7837"
        elif r["config"].startswith("combined"): c = "#7b3294"
        elif r["config"].startswith("baseline"): c = "#333333"
        colors.append(c)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(range(len(show)), rec, color=colors, width=.7)
    ax.axhline(base_recall, color="#333333", ls="--", lw=1.2,
               label=f"predict_tile baseline R={base_recall:.3f}")
    ax.axhline(ev_recall, color="#888888", ls=":", lw=1.2,
               label=f"committed evaluate baseline R={ev_recall:.3f}")
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("recall (IoU 0.4)")
    ax.set_title("s09 — recall by config (NE test AOI) vs baseline")
    ax.legend(loc="upper left", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "recall_exp" / "recall_by_config.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _summary(rows, base, evP, evR):
    pt = [r for r in rows if r["method"] == "predict_tile"]
    def best(prefix):
        c = [r for r in pt if r["config"].startswith(prefix)]
        return max(c, key=lambda r: r["recall"]) if c else None
    bA, bB, bC = best("A_"), best("B_"), best("C_")
    comb = next((r for r in pt if r["config"] == "combined_best"), None)
    b = base["recall"]
    print("\n========================= SUMMARY =========================")
    print(f"committed baseline (evaluate, per-tile) : P={evP:.3f} R={evR:.3f}  [integrity OK]")
    print(f"predict_tile baseline (whole-AOI anchor): P={base['precision']:.3f} "
          f"R={base['recall']:.3f} F1={base['f1']:.3f} pred={base['predicted_count']}")
    def line(name, r):
        if not r: return
        dr = r["recall"] - b
        print(f"  best {name:11s}: {r['config']:20s} R={r['recall']:.3f} "
              f"({dr:+.3f}) P={r['precision']:.3f} pred={r['predicted_count']}")
    line("threshold(A)", bA); line("tiling(B)", bB); line("resolution(C)", bC)
    line("combined", comb)
    movers = [(n, r) for n, r in [("threshold", bA), ("tiling", bB), ("resolution", bC)] if r]
    movers.sort(key=lambda t: -(t[1]["recall"] - b))
    if movers:
        top = movers[0]
        print(f"\n==> biggest recall lever: {top[0]} "
              f"(+{top[1]['recall']-b:.3f} recall, precision {top[1]['precision']:.3f} "
              f"vs {base['precision']:.3f})")
    print("NOTE: lowered precision is partly the INVENTORY-COVERAGE GAP — the ortho")
    print("contains real trees absent from the inventory, so genuine detections over")
    print("those crowns are scored as false positives. Precision here is a LOWER bound.")
    print("===========================================================")


if __name__ == "__main__":
    main()
