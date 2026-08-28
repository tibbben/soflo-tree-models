"""
Step 16 — BCNP YOLO detection (partner's Gables approach, adapted).  [local, heavy]

Adapts origin/ahsan's YOLO pipeline (READ-ONLY reference) to BCNP and to OUR honest
eval. Adopts the portable Gables wins, re-tuned for BCNP's smaller/denser wetland
crowns; scored by DISTANCE at TIGHT radii with a random-scatter floor (NOT their 5 m).

  stage 1  chip plots at a constant GROUND footprint (gsd = variable), fixed-size
           boxes on census ALIVE points, background negatives; clip to plot polygon
           + valid pixels. Lazy windowed reads, chips cached to disk.  <- always
  stage 2  resolution sweep (gsd 3 cm vs 5 cm @ fixed box) then box-size sweep
           (1.5/2/2.5/3 m @ best gsd); pick the config with best val-plot F1 @1.5 m
  stage 3  final long YOLO train at the winning (gsd, box) — aerial aug, early stop
  stage 4  honest DISTANCE eval on the held-out test plots: box centroids -> nearest
           census point, greedy highest-conf-first, one-to-one, KD-tree, clipped to
           polygon; P/R/F1 at 1.0 AND 1.5 m + a RANDOM-scatter floor (model-minus-
           random is the real number); predicted-vs-census counts per plot
  stage 5  figures (predictions-vs-census overlay, PR-vs-threshold, training curve,
           sweep results) + verdict vs s15 and the random floor

Model: YOLO26s (partner's champion) if resolvable, else YOLOv8s. COCO-pretrained.
Frugal for an 18 GB Mac: device=mps, cache=False (no images in RAM), workers=2,
batch=4, amp off (unstable on MPS), imgsz 640, never a whole 9500x9500 tif in RAM.
Peak RAM printed. Test split = plot_9_3 + plot_12_1 (same as s15 -> comparable).

Writes only under outputs/bcnp_yolo + reports/bcnp_yolo. Chips + weights gitignored.

Run:  .venv/bin/python src/s16_bcnp_yolo.py
      TD_SKIP_SWEEP=1 ...  # reuse the recorded best config, skip the sweep
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
import json
import time
import shutil
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box as shp_box
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

GT_COLOR = "#39ff14"
PRED_COLOR = "#d73027"
SEED = 0


def bp(cfg):
    return cfg["bcnp_yolo"]


def bc_dir(cfg):
    return (Path("~/Downloads") / bp(cfg).get("data_subdir", "bcnp")).expanduser()


def paths(cfg):
    od = C.p(cfg, cfg["outputs_dir"]) / "bcnp_yolo"
    rd = C.p(cfg, cfg["reports_dir"]) / "bcnp_yolo"
    (od / "chips").mkdir(parents=True, exist_ok=True)
    (od / "runs").mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True, exist_ok=True)
    return od, rd


def peak_ram_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def _plot_uid(name):
    u, up = name.split("_")[1:]
    return int(u), int(up)


def census_world(cfg, name, census, polys):
    """World (x,y) EPSG:32617 for ALIVE census trees in a plot."""
    u, up = _plot_uid(name)
    sub = census[(census.UNIT == u) & (census.UNITPLOT == up) & (census.ALIVE == 1)].dropna(subset=["XCOORD", "YCOORD"])
    poly = polys[polys.Name == name].to_crs(cfg["crs"]).geometry.iloc[0]
    swx, swy = poly.bounds[0], poly.bounds[1]
    wx = swx + pd.to_numeric(sub.XCOORD, errors="coerce").values
    wy = swy + pd.to_numeric(sub.YCOORD, errors="coerce").values
    return np.stack([wx, wy], 1), poly


# ── stage 1: constant-ground-footprint chipping (lazy windowed reads) ─────────
def _offsets(total, size, step):
    offs = list(range(0, max(1, total - size), step))
    if not offs or offs[-1] != total - size:
        offs.append(max(0, total - size))
    return sorted(set(offs))


def chip_plot(cfg, name, ground_m, radius_m, img_dir, lbl_dir, split,
              census, polys, want_labels=True, bg_ratio=3, rng=None):
    """Chip one plot at a constant ground footprint. Returns per-chip georef metadata
    (for mapping detections back to world). Writes PNG (+ YOLO txt if want_labels)."""
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from affine import Affine
    cf = bp(cfg)
    out_size = int(cf["out_size"])
    overlap = float(cf["overlap_frac"])
    rng = rng or np.random.default_rng(SEED)

    gt_xy, poly = census_world(cfg, name, census, polys)
    tif = bc_dir(cfg) / f"{name}.tif"
    img_dir = Path(img_dir); lbl_dir = Path(lbl_dir)
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    box_out_px = radius_m / ground_m * out_size

    meta, n_tree_chips, n_bg, n_boxes = [], 0, 0, 0
    with rasterio.open(tif) as src:
        tf = src.transform; gsd = abs(tf.a)
        src_chip = int(round(ground_m / gsd))
        step = max(1, int(round(src_chip * (1 - overlap))))
        from rasterio.transform import rowcol
        rr, cc = rowcol(tf, gt_xy[:, 0], gt_xy[:, 1])
        px_x = np.asarray(cc, float); px_y = np.asarray(rr, float)
        minx, miny, maxx, maxy = poly.bounds
        r_top, c_left = rowcol(tf, minx, maxy); r_bot, c_right = rowcol(tf, maxx, miny)
        r0, r1 = sorted((r_top, r_bot)); c0, c1 = sorted((c_left, c_right))
        r0 = max(0, r0); c0 = max(0, c0); r1 = min(src.height, r1); c1 = min(src.width, c1)

        bg_pending = []
        for row0 in _offsets(r1 - r0, src_chip, step):
            for col0 in _offsets(c1 - c0, src_chip, step):
                ro, co = r0 + row0, c0 + col0
                win = Window(co, ro, src_chip, src_chip)
                arr = src.read((1, 2, 3, 4) if src.count >= 4 else (1, 2, 3),
                               window=win, boundless=True, fill_value=0,
                               out_shape=((4 if src.count >= 4 else 3), out_size, out_size),
                               resampling=Resampling.bilinear)
                rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)
                alpha = arr[3] if arr.shape[0] >= 4 else np.full((out_size, out_size), 255, np.uint8)
                win_t = src.window_transform(win)
                out_t = win_t * Affine.scale(src_chip / out_size)
                inside = geometry_mask([poly], out_shape=(out_size, out_size), transform=out_t, invert=True)
                valid = inside & (alpha > 0) & (rgb.max(2) > 8)
                if valid.mean() < 0.5:
                    continue
                rgb = rgb * valid[..., None]
                left = win_t.c; top = win_t.f
                # census points in this window -> output px
                m = ((px_x >= co) & (px_x < co + src_chip) & (px_y >= ro) & (px_y < ro + src_chip))
                labels = []
                for xx, yy in zip(px_x[m], px_y[m]):
                    ox = (xx - co) / src_chip * out_size
                    oy = (yy - ro) / src_chip * out_size
                    if not (0 <= int(oy) < out_size and 0 <= int(ox) < out_size) or not valid[int(oy), int(ox)]:
                        continue
                    xmin = max(0, ox - box_out_px); ymin = max(0, oy - box_out_px)
                    xmax = min(out_size, ox + box_out_px); ymax = min(out_size, oy + box_out_px)
                    if xmax - xmin < 2 or ymax - ymin < 2:
                        continue
                    labels.append(f"0 {(xmin+xmax)/2/out_size:.6f} {(ymin+ymax)/2/out_size:.6f} "
                                  f"{(xmax-xmin)/out_size:.6f} {(ymax-ymin)/out_size:.6f}")
                rec = dict(left=left, top=top, ground_m=ground_m)
                if labels or not want_labels:
                    tile = f"{name}_r{ro}_c{co}.png"
                    Image.fromarray(rgb).save(img_dir / tile)
                    if want_labels:
                        (lbl_dir / tile.replace(".png", ".txt")).write_text("\n".join(labels))
                    rec["tile"] = tile; meta.append(rec)
                    n_tree_chips += bool(labels); n_boxes += len(labels)
                elif want_labels:
                    bg_pending.append((rgb, f"{name}_r{ro}_c{co}.png"))
        # subsample background negatives (train only)
        if want_labels and bg_pending and n_tree_chips:
            keep = min(len(bg_pending), max(0, n_tree_chips // bg_ratio))
            for i in rng.choice(len(bg_pending), keep, replace=False) if keep else []:
                rgb, tile = bg_pending[i]
                Image.fromarray(rgb).save(img_dir / tile)
                (lbl_dir / tile.replace(".png", ".txt")).write_text("")
                n_bg += 1
    return meta, dict(plot=name, split=split, gsd_cm=round(gsd * 100, 2),
                      census=len(gt_xy), tree_chips=n_tree_chips, bg_chips=n_bg, boxes=n_boxes)


def build_dataset(cfg, gsd_m, radius_m, census, polys, tag):
    """Chip train plots -> train split, val plot -> val split. Returns dataset.yaml path."""
    cf = bp(cfg)
    od, _ = paths(cfg)
    ground_m = gsd_m * int(cf["out_size"])
    root = od / "chips" / tag
    ds_yaml = root / "dataset.yaml"
    if ds_yaml.exists() and (root / "images" / "train").exists():
        return ds_yaml, None            # reuse cached chips
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(SEED)
    stats = []
    for name in cf["train_plots"]:
        _, st = chip_plot(cfg, name, ground_m, radius_m, root / "images/train", root / "labels/train",
                          "train", census, polys, want_labels=True, bg_ratio=int(cf["bg_ratio"]), rng=rng)
        stats.append(st)
    _, st = chip_plot(cfg, cf["val_plot"], ground_m, radius_m, root / "images/val", root / "labels/val",
                      "val", census, polys, want_labels=True, bg_ratio=int(cf["bg_ratio"]), rng=rng)
    stats.append(st)
    ds_yaml.write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\"Tree\"]\n")
    return ds_yaml, stats


# ── YOLO train / predict ─────────────────────────────────────────────────────
def _resolve_model(cfg):
    from ultralytics import YOLO
    for name in (bp(cfg)["model"], "yolov8s.pt"):
        try:
            YOLO(name)
            return name
        except Exception as e:
            print(f"   model {name} unavailable ({str(e)[:60]}); trying fallback")
    raise RuntimeError("no YOLO weights resolvable")


def train_yolo(cfg, ds_yaml, epochs, patience, run_name, model_name):
    from ultralytics import YOLO
    cf = bp(cfg)
    od, _ = paths(cfg)
    device = "mps"
    m = YOLO(model_name)
    m.train(data=str(ds_yaml), epochs=int(epochs), patience=int(patience),
            imgsz=int(cf["imgsz"]), batch=int(cf["batch"]), device=device,
            workers=int(cf["workers"]), amp=bool(cf["amp"]), cache=bool(cf["cache"]),
            project=str(od / "runs"), name=run_name, exist_ok=True, verbose=False,
            plots=False, seed=SEED, **cf["aug"])
    best = od / "runs" / run_name / "weights" / "best.pt"
    return best


# ── detection helpers (distance eval, mirrors s15) ───────────────────────────
def _nms_world(pts, scores, min_dist_m):
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    from scipy.spatial import cKDTree
    order = np.argsort(-scores); pts = pts[order]; sc = scores[order]
    keep = np.ones(len(pts), bool); tree = cKDTree(pts)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(pts[i], min_dist_m):
            if j > i and keep[j]:
                keep[j] = False
    return pts[keep], sc[keep]


def match_distance(pred_xy, pred_sc, gt_xy, radius):
    n_pred, n_gt = len(pred_xy), len(gt_xy)
    if n_pred == 0 or n_gt == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, tp=0, n_pred=n_pred, n_gt=n_gt)
    from scipy.spatial import cKDTree
    order = np.argsort(-np.asarray(pred_sc)); tree = cKDTree(gt_xy)
    used, tp = set(), 0
    for i in order:
        d, j = tree.query(pred_xy[i])
        if d <= radius and j not in used:
            used.add(j); tp += 1
    prec, rec = tp / n_pred, tp / n_gt
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, n_pred=n_pred, n_gt=n_gt)


def random_baseline(cfg, per_plot, polys, radii, trials=8):
    rng = np.random.default_rng(SEED); out = {}
    for radius in radii:
        Ps, Rs, F = [], [], []
        for _ in range(trials):
            pw_all, gts = [], []
            for name, info in per_plot.items():
                npred = len(info["pred"]); gt = info["gt"]
                poly = polys[polys.Name == name].to_crs(cfg["crs"]).geometry.iloc[0]
                minx, miny, maxx, maxy = poly.bounds
                pw_all.append(np.stack([rng.uniform(minx, maxx, npred), rng.uniform(miny, maxy, npred)], 1))
                gts.append(gt)
            pw = np.vstack(pw_all) if pw_all else np.zeros((0, 2)); gt = np.vstack(gts)
            r = match_distance(pw, rng.random(len(pw)), gt, radius)
            Ps.append(r["precision"]); Rs.append(r["recall"]); F.append(r["f1"])
        out[radius] = dict(precision=float(np.mean(Ps)), recall=float(np.mean(Rs)), f1=float(np.mean(F)))
    return out


def predict_plot_raw(cfg, name, best_pt, gsd_m, census, polys, conf_floor=0.02):
    """Chip a plot (images only), YOLO-predict each chip, return raw world detections
    (centroids) with scores, clipped to the plot polygon. Frugal: one chip at a time."""
    from ultralytics import YOLO
    cf = bp(cfg); od, _ = paths(cfg)
    ground_m = gsd_m * int(cf["out_size"]); out_size = int(cf["out_size"])
    tmp = od / "chips" / f"_pred_{name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    meta, _ = chip_plot(cfg, name, ground_m, 2.0, tmp / "images", tmp / "labels",
                        "pred", census, polys, want_labels=False)
    _, poly = census_world(cfg, name, census, polys)
    model = YOLO(str(best_pt))
    world, score = [], []
    for rec in meta:
        res = model.predict(str(tmp / "images" / rec["tile"]), imgsz=int(cf["imgsz"]),
                            conf=conf_floor, device="mps", verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xywh = res.boxes.xywh.cpu().numpy(); conf = res.boxes.conf.cpu().numpy()
        for (cx, cy, w, h), s in zip(xywh, conf):
            wx = rec["left"] + cx / out_size * ground_m
            wy = rec["top"] - cy / out_size * ground_m
            from shapely.geometry import Point
            if poly.contains(Point(wx, wy)):
                world.append((wx, wy)); score.append(float(s))
    shutil.rmtree(tmp, ignore_errors=True)
    return (np.array(world).reshape(-1, 2), np.array(score))


def best_f1_over_conf(cfg, raw_xy, raw_sc, gt, radius):
    """Best F1 across the conf sweep at a fixed match radius (for sweep selection)."""
    cf = bp(cfg); best = dict(f1=-1)
    for thr in cf["conf_sweep"]:
        m = raw_sc >= thr
        pw, ps = _nms_world(raw_xy[m], raw_sc[m], cf["peak_min_dist_m"])
        r = match_distance(pw, ps, gt, radius)
        if r["f1"] > best["f1"]:
            best = dict(conf=thr, **r)
    return best


# ── stage 2: sweeps ──────────────────────────────────────────────────────────
def run_sweep(cfg, od, rd, census, polys, model_name):
    cf = bp(cfg)
    val = cf["val_plot"]; sel_r = float(cf["select_radius_m"])
    gt_val, _ = census_world(cfg, val, census, polys)
    rows = []

    def try_config(gsd, box, tag):
        ds, st = build_dataset(cfg, gsd, box, census, polys, tag)
        if st is not None:
            tt = sum(s["tree_chips"] for s in st if s["split"] == "train")
            print(f"   [{tag}] chips: train tree={tt} "
                  f"bg={sum(s['bg_chips'] for s in st if s['split']=='train')} "
                  f"val_tree={sum(s['tree_chips'] for s in st if s['split']=='val')}", flush=True)
        t0 = time.time()
        best_pt = train_yolo(cfg, ds, cf["sweep_epochs"], cf["sweep_patience"], f"sweep_{tag}", model_name)
        rx, rs = predict_plot_raw(cfg, val, best_pt, gsd, census, polys)
        b = best_f1_over_conf(cfg, rx, rs, gt_val, sel_r)
        print(f"   [{tag}] gsd={gsd*100:.0f}cm box={box}m -> val F1@{sel_r}m={b['f1']:.3f} "
              f"(P{b['precision']:.2f}/R{b['recall']:.2f} conf{b.get('conf')}) {time.time()-t0:.0f}s "
              f"RAM{peak_ram_gb():.1f}GB", flush=True)
        rows.append(dict(phase="gsd" if box == cf["sweep_box_radius_m"] else "box",
                         gsd_cm=round(gsd * 100, 1), box_m=box, val_f1=round(b["f1"], 4),
                         precision=round(b["precision"], 4), recall=round(b["recall"], 4), conf=b.get("conf")))
        return b["f1"]

    print("\n[stage2] GSD sweep (fixed box "
          f"{cf['sweep_box_radius_m']} m) ...", flush=True)
    gsd_scores = {}
    for gsd in cf["gsd_sweep_m"]:
        tag = f"gsd{int(gsd*100)}_box{cf['sweep_box_radius_m']}".replace(".", "p")
        gsd_scores[gsd] = try_config(gsd, cf["sweep_box_radius_m"], tag)
    best_gsd = max(gsd_scores, key=gsd_scores.get)
    print(f"[stage2] best GSD = {best_gsd*100:.0f} cm", flush=True)

    print(f"\n[stage2] box-size sweep at {best_gsd*100:.0f} cm ...", flush=True)
    box_scores = {cf["sweep_box_radius_m"]: gsd_scores[best_gsd]}   # reuse from GSD sweep
    for box in cf["box_sweep_radius_m"]:
        if box == cf["sweep_box_radius_m"]:
            continue
        tag = f"gsd{int(best_gsd*100)}_box{box}".replace(".", "p")
        box_scores[box] = try_config(best_gsd, box, tag)
    best_box = max(box_scores, key=box_scores.get)
    print(f"[stage2] best box = {best_box} m  (val F1 {box_scores[best_box]:.3f})", flush=True)

    with open(od / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "gsd_cm", "box_m", "val_f1", "precision", "recall", "conf"])
        w.writeheader(); w.writerows(rows)
    (od / "best_config.json").write_text(json.dumps(dict(gsd_m=best_gsd, box_m=best_box)))
    _fig_sweep(rd, rows, best_gsd, best_box)
    return best_gsd, best_box, rows


# ── stage 4: honest eval ─────────────────────────────────────────────────────
def evaluate(cfg, od, rd, best_pt, gsd_m, census, polys):
    cf = bp(cfg)
    raw = {}
    for name in cf["test_plots"]:
        rx, rs = predict_plot_raw(cfg, name, best_pt, gsd_m, census, polys)
        raw[name] = dict(rx=rx, rs=rs)
        print(f"   {name}: {len(rx)} raw detections (conf>=0.02, "
              f"score max {rs.max():.3f})" if len(rs) else f"   {name}: 0 detections", flush=True)

    sweep = _threshold_sweep(cfg, od, raw, census, polys)
    ec = cf["eval_conf"]
    if isinstance(ec, str) and ec.strip().lower() == "auto":
        conf = float(max(sweep, key=lambda s: s["f1"])["conf"]) if sweep else 0.1
        print(f"[stage4] auto eval_conf = {conf} (best sweep F1)", flush=True)
    else:
        conf = float(ec)

    per_plot = {}
    for name in cf["test_plots"]:
        m = raw[name]["rs"] >= conf
        pw, ps = _nms_world(raw[name]["rx"][m], raw[name]["rs"][m], cf["peak_min_dist_m"])
        per_plot[name] = dict(pred=pw, score=ps, gt=census_world(cfg, name, census, polys)[0])

    rows = []
    print(f"\n===== STAGE 4 — distance eval (conf={conf}) =====")
    hdr = f"{'plot':11s} {'radius':>6} {'pred':>5} {'census':>6} {'TP':>5} {'prec':>6} {'recall':>7} {'F1':>6}"
    print(hdr); print("-" * len(hdr))
    all_pred = np.vstack([per_plot[n]["pred"] for n in cf["test_plots"] if len(per_plot[n]["pred"])]) if any(len(per_plot[n]["pred"]) for n in cf["test_plots"]) else np.zeros((0, 2))
    all_sc = np.concatenate([per_plot[n]["score"] for n in cf["test_plots"] if len(per_plot[n]["score"])]) if any(len(per_plot[n]["score"]) for n in cf["test_plots"]) else np.zeros(0)
    all_gt = np.vstack([per_plot[n]["gt"] for n in cf["test_plots"]])
    scopes = [(n, per_plot[n]["pred"], per_plot[n]["score"], per_plot[n]["gt"]) for n in cf["test_plots"]]
    scopes.append(("ALL", all_pred, all_sc, all_gt))
    for nm, pw, ps, gt in scopes:
        for radius in cf["match_radii_m"]:
            r = match_distance(pw, ps, gt, radius)
            rows.append(dict(plot=nm, radius_m=radius, n_pred=r["n_pred"], n_census=r["n_gt"], tp=r["tp"],
                             precision=round(r["precision"], 4), recall=round(r["recall"], 4), f1=round(r["f1"], 4)))
            print(f"{nm:11s} {radius:>6} {r['n_pred']:>5} {r['n_gt']:>6} {r['tp']:>5} "
                  f"{r['precision']:>6.3f} {r['recall']:>7.3f} {r['f1']:>6.3f}")

    rnd = random_baseline(cfg, per_plot, polys, cf["match_radii_m"])
    print("\n  random-scatter baseline (same counts, uniform in plot):")
    for rad in cf["match_radii_m"]:
        rb = rnd[rad]
        print(f"    radius {rad} m: P={rb['precision']:.3f} R={rb['recall']:.3f} F1={rb['f1']:.3f}")
    _write_eval(od, rows, rnd)
    _fig_pr_sweep(rd, sweep, cf["match_radii_m"])
    _fig_overlay(cfg, od, rd, per_plot)
    return rows, per_plot, sweep, rnd


def _threshold_sweep(cfg, od, raw, census, polys):
    cf = bp(cfg); radius = max(cf["match_radii_m"])
    gt = np.vstack([census_world(cfg, n, census, polys)[0] for n in cf["test_plots"]])
    out = []
    for thr in cf["conf_sweep"]:
        pw_all, ps_all = [], []
        for name in cf["test_plots"]:
            m = raw[name]["rs"] >= thr
            pw, ps = _nms_world(raw[name]["rx"][m], raw[name]["rs"][m], cf["peak_min_dist_m"])
            if len(pw):
                pw_all.append(pw); ps_all.append(ps)
        pw = np.vstack(pw_all) if pw_all else np.zeros((0, 2))
        ps = np.concatenate(ps_all) if ps_all else np.zeros(0)
        r = match_distance(pw, ps, gt, radius)
        out.append(dict(conf=thr, radius_m=radius, n_pred=len(pw), tp=r["tp"],
                        precision=round(r["precision"], 4), recall=round(r["recall"], 4), f1=round(r["f1"], 4)))
        print(f"   sweep conf={thr:<4} n_pred={len(pw):5d} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    with open(od / "pr_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conf", "radius_m", "n_pred", "tp", "precision", "recall", "f1"])
        w.writeheader(); w.writerows(out)
    return out


def _write_eval(od, rows, rnd):
    fields = ["plot", "radius_m", "n_pred", "n_census", "tp", "precision", "recall", "f1"]
    out = list(rows)
    for rad, rb in rnd.items():
        out.append(dict(plot="RANDOM", radius_m=rad, n_pred="", n_census="", tp="",
                        precision=round(rb["precision"], 4), recall=round(rb["recall"], 4), f1=round(rb["f1"], 4)))
    with open(od / "eval_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    print(f"\nwrote {od/'eval_metrics.csv'}")


# ── figures ──────────────────────────────────────────────────────────────────
def _fig_sweep(rd, rows, best_gsd, best_box):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    gsd_rows = [r for r in rows if r["phase"] == "gsd"]
    box_rows = [r for r in rows if r["phase"] == "box" or (r["phase"] == "gsd" and abs(r["gsd_cm"] - best_gsd*100) < 0.1)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    if gsd_rows:
        a1.plot([r["gsd_cm"] for r in gsd_rows], [r["val_f1"] for r in gsd_rows], "-o", color="#1b7837")
    a1.set_xlabel("working GSD (cm/px)"); a1.set_ylabel(f"val F1"); a1.set_title("s16 — resolution sweep")
    bx = sorted({(r["box_m"], r["val_f1"]) for r in box_rows})
    if bx:
        a2.plot([b[0] for b in bx], [b[1] for b in bx], "-o", color="#4575b4")
    a2.axvline(best_box, color="#d73027", ls="--", lw=1, label=f"best {best_box} m")
    a2.set_xlabel("box radius (m)"); a2.set_ylabel("val F1")
    a2.set_title(f"s16 — box-size sweep @ {best_gsd*100:.0f} cm"); a2.legend()
    fig.tight_layout(); fig.savefig(rd/"sweep_results.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'sweep_results.png'}")


def _fig_pr_sweep(rd, sweep, radii):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    conf = [s["conf"] for s in sweep]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(conf, [s["precision"] for s in sweep], "-o", color="#4575b4", ms=4, label="precision")
    ax.plot(conf, [s["recall"] for s in sweep], "-o", color="#d73027", ms=4, label="recall")
    ax.plot(conf, [s["f1"] for s in sweep], "-o", color="#7b3294", ms=4, label="F1")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel(f"score (match radius {max(radii)} m)")
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("s16 — precision / recall / F1 vs threshold")
    fig.tight_layout(); fig.savefig(rd/"pr_sweep.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'pr_sweep.png'}")


def _fig_overlay(cfg, od, rd, per_plot):
    import rasterio
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    name = bp(cfg)["test_plots"][0]; info = per_plot[name]
    with rasterio.open(bc_dir(cfg) / f"{name}.tif") as src:
        tf = src.transform
        step = max(1, int(round(max(src.width, src.height) / 1600)))
        img = src.read([1, 2, 3], out_shape=(3, src.height // step, src.width // step))
        H, W = img.shape[1], img.shape[2]

    def to_px(x, y):
        return (x - tf.c) / tf.a / step, (y - tf.f) / tf.e / step
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(np.transpose(img, (1, 2, 0)))
    gx = [to_px(x, y) for x, y in info["gt"]]; px = [to_px(x, y) for x, y in info["pred"]]
    if gx:
        ax.scatter([p[0] for p in gx], [p[1] for p in gx], s=10, c=GT_COLOR, marker="+", linewidths=0.6, label=f"census ({len(gx)})")
    if px:
        ax.scatter([p[0] for p in px], [p[1] for p in px], s=14, facecolors="none", edgecolors=PRED_COLOR, linewidths=0.6, label=f"YOLO ({len(px)})")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.legend(loc="upper right", fontsize=9, markerscale=2)
    ax.set_title(f"s16 — YOLO detections vs census on held-out {name}")
    fig.tight_layout(); fig.savefig(rd/f"overlay_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/f'overlay_{name}.png'}")


def _fig_curve(cfg, od, rd, run_name):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    res = od / "runs" / run_name / "results.csv"
    if not res.exists():
        return
    df = pd.read_csv(res); df.columns = [c.strip() for c in df.columns]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for col, color in [("train/box_loss", "#1b7837"), ("val/box_loss", "#d73027")]:
        if col in df:
            a1.plot(df["epoch"], df[col], "-", color=color, label=col.split("/")[0])
    a1.set_xlabel("epoch"); a1.set_ylabel("box loss"); a1.legend(); a1.set_title("s16 — loss")
    for col, color, lbl in [("metrics/precision(B)", "#4575b4", "precision"),
                            ("metrics/recall(B)", "#d73027", "recall"), ("metrics/mAP50(B)", "#7b3294", "mAP50")]:
        if col in df:
            a2.plot(df["epoch"], df[col], "-", color=color, label=lbl)
    a2.set_xlabel("epoch"); a2.set_ylabel("val metric (YOLO IoU-based)"); a2.legend()
    a2.set_title("s16 — YOLO val metrics")
    fig.suptitle("s16 — final YOLO training (BCNP)", fontsize=12)
    fig.tight_layout(); fig.savefig(rd/"train_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'train_curve.png'}")


def verdict(cfg, eval_rows, rnd, best_gsd, best_box):
    allr = {r["radius_m"]: r for r in eval_rows if r["plot"] == "ALL"}
    print("\n========================= VERDICT =========================")
    print(f"best config: GSD {best_gsd*100:.0f} cm/px, fixed box radius {best_box} m")
    for rad in sorted(allr):
        r = allr[rad]; rb = rnd[rad]
        print(f"radius {rad} m:  MODEL R {r['recall']*100:4.1f}% P {r['precision']*100:4.1f}% "
              f"F1 {r['f1']*100:4.1f}%  |  RANDOM R {rb['recall']*100:4.1f}% P {rb['precision']*100:4.1f}%  "
              f"(edge {(r['recall']-rb['recall'])*100:+.1f} pts R)  [pred {r['n_pred']} vs census {r['n_census']}]")
    tight = min(allr); m_t, r_t = allr[tight], rnd[tight]
    beats = m_t["f1"] > r_t["f1"] and m_t["precision"] > r_t["precision"]
    print(f"s15 heatmap peak-finder (same test, @1 m): F1 ~11%, edge over random +1.9 pts R")
    print(f"==> YOLO {'BEATS' if beats else 'matches'} random at {tight} m "
          f"(F1 {m_t['f1']*100:.1f}% vs {r_t['f1']*100:.1f}%, P {m_t['precision']*100:.1f}% vs {r_t['precision']*100:.1f}%).")
    print(f"peak RAM this process: {peak_ram_gb():.1f} GB")
    print("NOTE: distance eval at TIGHT radii + random floor (NOT partner's lenient 5 m);")
    print("census exhaustive only inside each 100 m plot polygon.")
    print("===========================================================")


def main():
    cfg = C.load_config(); np.random.seed(SEED)
    import torch; torch.manual_seed(SEED)
    od, rd = paths(cfg)
    bcd = bc_dir(cfg)
    census = pd.read_csv(bcd / "RP.Plot_census_data.2025.csv")
    polys = gpd.read_file(bcd / "tree_plots_polygons.geojson")
    model_name = _resolve_model(cfg)
    print(f"[stage0] model = {model_name}  device=mps  batch={bp(cfg)['batch']} "
          f"imgsz={bp(cfg)['imgsz']}  amp={bp(cfg)['amp']}  cache={bp(cfg)['cache']}", flush=True)

    skip = os.environ.get("TD_SKIP_SWEEP", "").strip().lower() in ("1", "true", "yes", "on")
    bestf = od / "best_config.json"
    if skip and bestf.exists():
        bc = json.loads(bestf.read_text()); best_gsd, best_box = bc["gsd_m"], bc["box_m"]
        print(f"[stage2] TD_SKIP_SWEEP — using recorded best GSD {best_gsd*100:.0f}cm box {best_box}m")
    else:
        best_gsd, best_box, _ = run_sweep(cfg, od, rd, census, polys, model_name)

    print(f"\n[stage3] final train @ GSD {best_gsd*100:.0f}cm box {best_box}m "
          f"(epochs {bp(cfg)['final_epochs']}, patience {bp(cfg)['final_patience']}) ...", flush=True)
    tag = f"final_gsd{int(best_gsd*100)}_box{best_box}".replace(".", "p")
    ds, _ = build_dataset(cfg, best_gsd, best_box, census, polys, tag)
    t0 = time.time()
    best_pt = train_yolo(cfg, ds, bp(cfg)["final_epochs"], bp(cfg)["final_patience"], "final", model_name)
    print(f"[stage3] final train done ({time.time()-t0:.0f}s, RAM {peak_ram_gb():.1f}GB) -> {best_pt}", flush=True)
    _fig_curve(cfg, od, rd, "final")

    try:
        eval_rows, per_plot, sweep, rnd = evaluate(cfg, od, rd, best_pt, best_gsd, census, polys)
        verdict(cfg, eval_rows, rnd, best_gsd, best_box)
    except Exception as e:
        import traceback
        print(f"\n[stage4] eval failed ({e}) — sweep + training artifacts stand.")
        traceback.print_exc()


if __name__ == "__main__":
    main()
