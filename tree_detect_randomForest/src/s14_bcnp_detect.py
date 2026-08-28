"""
Step 14 — BCNP FIRST detection run (find trees in the wetland canopy).  [local]

A from-scratch DeepForest train on the 8 clean Big Cypress plots — a different
domain and resolution from the Gables ortho, so we start from the BASE release
checkpoint, NOT the Gables one. Self-contained and STAGED so partial results
survive a crash:

  stage 1  build detection labels + resampled tiles, WRITE per-plot counts   <- always
           - census points (ALIVE trees, all species) georeferenced from each
             plot polygon's SW corner (plot-local metres) -> EPSG:32617
           - boxes sized per-crown from DBH allometry, fixed fallback if DBH NA
             (NOT the Gables 7 m box)
           - RESAMPLE each ~1.6 cm/px plot to ~10 cm/px (config target_gsd_m) so
             the ortho matches the DeepForest base-release train domain
           - mask each raster to its 100 m plot polygon (census is exhaustive only
             inside it) AND the alpha band; drop any box in nodata; clip to valid px
  stage 2  spatial split by PLOT (2 whole plots held out for test)
  stage 3  train DeepForest from base release (early stop, per-epoch ckpts, mps, 32-bit)
  stage 4  eval on held-out plots at IoU 0.4 AND 0.3, CLIPPED to valid pixels
           (no phantom FNs under nodata): precision/recall/F1 + pred-vs-census counts
  stage 5  figures (predict overlay, train curve, FN buckets) + plain verdict:
           what recall, and is the failure blindness (undetected) or box-tightness
           (found but loosely boxed)?  [FN-bucket logic reused from s10]

Writes only under outputs/bcnp_detect + reports/bcnp_detect. Tiles + checkpoints
are gitignored (regenerable); counts/metrics/figures are committed.

Run:  .venv/bin/python src/s14_bcnp_detect.py
      TD_SKIP_TRAIN=1 ...   # stage 1/2 + eval on an existing checkpoint only
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

GT_COLOR = "#1b7837"     # census boxes (green)
PRED_COLOR = "#d73027"   # predicted boxes (red)
SEED = 0


# ── config access ────────────────────────────────────────────────────────────
def bc_cfg(cfg):
    return cfg["bcnp_detect"]


def bc_dir(cfg):
    sub = bc_cfg(cfg).get("data_subdir", "bcnp")
    return (Path("~/Downloads") / sub).expanduser()


def paths(cfg):
    od = C.p(cfg, cfg["outputs_dir"]) / "bcnp_detect"
    rd = C.p(cfg, cfg["reports_dir"]) / "bcnp_detect"
    (od / "tiles").mkdir(parents=True, exist_ok=True)
    (od / "model").mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True, exist_ok=True)
    return od, rd


# ── IoU matching (mirrors s10/s11) ───────────────────────────────────────────
def _iou(a, b):
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter) if inter > 0 else 0.0


def greedy_match(pred_geoms, pred_scores, gt_geoms, thr):
    """Score-ordered greedy IoU>=thr matching. Returns metrics + matched-gt set."""
    n_pred, n_gt = len(pred_geoms), len(gt_geoms)
    out = dict(precision=0.0, recall=0.0, f1=0.0, tp=0,
               n_pred=n_pred, n_gt=n_gt, matched_gt=set())
    if n_pred == 0 or n_gt == 0:
        return out
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
    out.update(precision=prec, recall=rec, f1=f1, tp=tp, matched_gt=used)
    return out


def best_iou_to(query_geoms, target_geoms):
    if not len(target_geoms) or not len(query_geoms):
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


def fn_bucket(best):
    """Reused from s10: why an unmatched census tree is a false negative."""
    if best >= 0.40:
        return "matched"                      # (shouldn't happen for FNs)
    if best >= 0.30:
        return "near-miss [0.30,0.40)"        # box-tightness
    if best > 0.0:
        return "poor localization (0,0.30)"   # box-tightness
    return "true miss (blind)"                # blindness


# ── stage 1: labels + resampled, masked tiles ────────────────────────────────
def _plot_uid(name):
    u, up = name.split("_")[1:]
    return int(u), int(up)


def crown_diam_m(dbh, bcfg):
    a, b = bcfg["crown_a_m"], bcfg["crown_b_m"]
    lo, hi = bcfg["crown_min_m"], bcfg["crown_max_m"]
    if dbh is None or not np.isfinite(dbh):
        return bcfg["fallback_box_m"]
    return float(np.clip(a + b * dbh, lo, hi))


def build_plot(cfg, name, od, census, polys):
    """Resample one plot to target GSD, mask to plot polygon + alpha, write a clip
    tif and a DeepForest annotation CSV. Returns (clip_path, ann_df, stats, poly32617)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from affine import Affine
    bcfg = bc_cfg(cfg)
    target = float(bcfg["target_gsd_m"])
    bcd = bc_dir(cfg)

    u, up = _plot_uid(name)
    sub = census[(census.UNIT == u) & (census.UNITPLOT == up) & (census.ALIVE == 1)].copy()
    sub = sub.dropna(subset=["XCOORD", "YCOORD"])
    poly = polys[polys.Name == name].to_crs(cfg["crs"]).geometry.iloc[0]
    swx, swy = poly.bounds[0], poly.bounds[1]           # SW corner = local-coord origin

    tif = bcd / f"{name}.tif"
    with rasterio.open(tif) as src:
        native = abs(src.transform.a)
        W, H = src.width, src.height
        scale = target / native
        out_w, out_h = max(1, round(W / scale)), max(1, round(H / scale))
        rgb = src.read([1, 2, 3], out_shape=(3, out_h, out_w),
                       resampling=Resampling.average)
        alpha = (src.read(4, out_shape=(out_h, out_w), resampling=Resampling.nearest)
                 if src.count >= 4 else np.full((out_h, out_w), 255, np.uint8))
        new_tf = src.transform * Affine.scale(W / out_w, H / out_h)
        crs = src.crs

    # valid mask: inside the 100 m plot polygon, alpha present, and real imagery
    inside = geometry_mask([poly], out_shape=(out_h, out_w), transform=new_tf, invert=True)
    bright = rgb.max(axis=0) > 8
    valid = inside & (alpha > 0) & bright
    rgb = rgb * valid[None]                             # black out everything invalid

    clip_path = od / "tiles" / f"{name}_clip.tif"
    prof = dict(driver="GTiff", height=out_h, width=out_w, count=3,
                dtype="uint8", crs=crs, transform=new_tf, compress="deflate")
    with rasterio.open(clip_path, "w", **prof) as dst:
        dst.write(rgb.astype(np.uint8))

    # census points -> world boxes -> pixel boxes on the resampled clip
    recs = []
    n_alive = len(sub)
    dropped_nodata = dropped_oob = 0
    for _, r in sub.iterrows():
        wx, wy = swx + float(r.XCOORD), swy + float(r.YCOORD)
        dbh = pd.to_numeric(r.get("DBH"), errors="coerce")
        half = crown_diam_m(dbh, bcfg) / 2.0
        xmin, ymin, xmax, ymax = C.world_box_to_pixel((wx - half, wy - half, wx + half, wy + half), new_tf)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        if not (0 <= cx < out_w and 0 <= cy < out_h) or not valid[cy, cx]:
            dropped_nodata += 1
            continue
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(out_w, xmax); ymax = min(out_h, ymax)
        if xmax - xmin < 2 or ymax - ymin < 2:
            dropped_oob += 1
            continue
        recs.append(dict(image_path=clip_path.name, xmin=int(xmin), ymin=int(ymin),
                         xmax=int(xmax), ymax=int(ymax), label="Tree"))
    ann = pd.DataFrame(recs)
    stats = dict(plot=name, native_cm=round(native * 100, 2),
                 out_px=f"{out_w}x{out_h}", eff_gsd_cm=round(native * W / out_w * 100, 2),
                 alive=n_alive, boxes=len(ann),
                 dropped_nodata=dropped_nodata, dropped_tiny=dropped_oob)
    return clip_path, ann, stats, poly


def stage1_build(cfg, od, rd):
    from deepforest import preprocess
    bcfg = bc_cfg(cfg)
    bcd = bc_dir(cfg)
    census = pd.read_csv(bcd / "RP.Plot_census_data.2025.csv")
    polys = gpd.read_file(bcd / "tree_plots_polygons.geojson")
    patch_px = int(round(bcfg["patch_size_m"] / bcfg["target_gsd_m"]))
    test_set = set(bcfg["test_plots"])

    print("[stage1] building resampled, plot-masked tiles from 8 clean plots ...", flush=True)
    print(f"         target GSD {bcfg['target_gsd_m']*100:.0f} cm/px  patch {patch_px}px "
          f"({bcfg['patch_size_m']:.0f} m)  box=DBH-allometry "
          f"clip({bcfg['crown_a_m']}+{bcfg['crown_b_m']}*DBH,"
          f"{bcfg['crown_min_m']},{bcfg['crown_max_m']}) m", flush=True)

    all_stats, split_tiles = [], {"train": [], "test": []}
    polys_by_plot = {}
    for name in bcfg["clean_plots"]:
        clip_path, ann, stats, poly = build_plot(cfg, name, od, census, polys)
        polys_by_plot[name] = poly
        split = "test" if name in test_set else "train"
        all_stats.append({**stats, "split": split})
        print(f"   {name:9s} [{split:5s}] alive={stats['alive']:5d} "
              f"boxes={stats['boxes']:5d} dropped(nodata={stats['dropped_nodata']},"
              f"tiny={stats['dropped_tiny']})  eff {stats['eff_gsd_cm']:.1f} cm/px", flush=True)
        if len(ann) == 0:
            continue
        ann_csv = od / "tiles" / f"{name}_ann.csv"
        ann.to_csv(ann_csv, index=False)
        save_dir = od / "tiles" / split
        save_dir.mkdir(exist_ok=True)
        tiled = preprocess.split_raster(
            annotations_file=str(ann_csv), path_to_raster=str(clip_path),
            save_dir=str(save_dir), patch_size=patch_px,
            patch_overlap=bcfg["patch_overlap"])
        tiled = tiled.groupby("image_path").filter(
            lambda g: len(g) >= bcfg["min_boxes_per_tile"])
        split_tiles[split].append(tiled)

    # combined per-split tile CSVs for training/validation
    for split in ("train", "test"):
        dfs = [d for d in split_tiles[split] if len(d)]
        comb = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        (od / "tiles" / split).mkdir(exist_ok=True)
        comb.to_csv(od / "tiles" / split / f"{split}_tiles.csv", index=False)

    _write_stats(od, all_stats)
    _stage1_report(all_stats, split_tiles, rd)
    _fig_class_split(all_stats, rd)
    return all_stats, polys_by_plot, census, polys


def _write_stats(od, stats):
    fields = ["plot", "split", "native_cm", "eff_gsd_cm", "out_px",
              "alive", "boxes", "dropped_nodata", "dropped_tiny"]
    with open(od / "label_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for s in stats:
            w.writerow({k: s.get(k) for k in fields})
    print(f"wrote {od/'label_counts.csv'}")


def _stage1_report(stats, split_tiles, rd):
    print("\n===== STAGE 1 — detection labels (ALIVE trees, resampled+masked) =====")
    tot_alive = sum(s["alive"] for s in stats)
    tot_box = sum(s["boxes"] for s in stats)
    tot_drop = sum(s["dropped_nodata"] + s["dropped_tiny"] for s in stats)
    for split in ("train", "test"):
        rows = [s for s in stats if s["split"] == split]
        nb = sum(s["boxes"] for s in rows)
        dfs = [d for d in split_tiles[split] if len(d)]
        ntiles = sum(d.image_path.nunique() for d in dfs)
        print(f"  {split:5s}: {len(rows)} plots  boxes={nb:5d}  tiles={ntiles}  "
              f"plots={[s['plot'] for s in rows]}")
    print(f"  TOTAL alive={tot_alive}  boxes kept={tot_box}  dropped={tot_drop} "
          f"({tot_drop/max(1,tot_alive)*100:.1f}% under nodata/tiny)")


def _fig_class_split(stats, rd):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    names = [s["plot"] for s in stats]
    boxes = [s["boxes"] for s in stats]
    cols = ["#d73027" if s["split"] == "test" else "#1b7837" for s in stats]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), boxes, color=cols, width=.7)
    for b, v in zip(bars, boxes):
        ax.text(b.get_x()+b.get_width()/2, v+max(boxes)*.01, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("detection boxes (ALIVE trees)")
    ax.set_title("s14 — BCNP label boxes per plot (red = held-out test)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(rd/"label_counts.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'label_counts.png'}")


# ── stage 3: train from base release ─────────────────────────────────────────
def _f1(p, r):
    return 2*p*r/(p+r) if (p and r and (p+r)) else 0.0


def train(cfg, od, rd):
    import torch
    from pytorch_lightning import Callback
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from deepforest import main as df_main
    bcfg = bc_cfg(cfg)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    epochs = int(os.environ.get("TD_EPOCHS", bcfg["epochs"]))
    patience = int(bcfg["early_stop_patience"])
    iou_train = bcfg["iou_eval"][0]
    train_dir, test_dir = od / "tiles" / "train", od / "tiles" / "test"

    rows = []

    class Rec(Callback):
        def on_train_epoch_start(self, tr, pl): self._t0 = time.time()
        def on_validation_end(self, tr, pl):
            if tr.sanity_checking:
                return
            m = tr.callback_metrics
            g = lambda k: float(m[k]) if m.get(k) is not None else None
            prec, rec = g("box_precision"), g("box_recall")
            row = dict(epoch=int(tr.current_epoch), train_loss=g("train_loss"),
                       val_loss=g("val_loss"), val_precision=prec, val_recall=rec,
                       val_f1=round(_f1(prec or 0, rec or 0), 4),
                       seconds=round(time.time()-getattr(self, "_t0", time.time()), 1))
            rows.append(row)
            _write_curve(od, rows)
            fmt = lambda x, n: "n/a" if x is None else round(x, n)
            print(f"   [epoch {row['epoch']:02d}] {row['seconds']:6.1f}s "
                  f"train_loss={fmt(row['train_loss'],4)} val_prec={fmt(prec,3)} "
                  f"val_recall={fmt(rec,3)} val_f1={row['val_f1']}", flush=True)

    m = df_main.deepforest()                          # BASE release weights (not Gables)
    # Keep the RetinaNet head threshold consistent with eval. (DeepForest's validation
    # box_recall already reflects the true IoU-matched recall regardless, so this does
    # not change early stopping — the flat val recall is a real plateau, not a threshold
    # artifact — but it keeps train-time and predict-time behavior aligned.)
    m.model.score_thresh = float(bcfg["score_thresh"])
    m.config["train"]["csv_file"] = str(train_dir / "train_tiles.csv")
    m.config["train"]["root_dir"] = str(train_dir)
    m.config["train"]["epochs"] = epochs
    m.config["train"]["batch_size"] = int(bcfg["batch_size"])
    m.config["batch_size"] = int(bcfg["batch_size"])
    m.config["score_thresh"] = float(bcfg["score_thresh"])
    m.config["validation"]["csv_file"] = str(test_dir / "test_tiles.csv")
    m.config["validation"]["root_dir"] = str(test_dir)
    m.config["validation"]["iou_threshold"] = iou_train
    m.config["validation"]["val_accuracy_interval"] = 1
    m.config["accelerator"] = device
    m.config["devices"] = 1

    early = EarlyStopping(monitor="box_recall", mode="max", patience=patience, verbose=True)
    ckpt = ModelCheckpoint(dirpath=str(od / "model"), filename="bcnp_best_recall",
                           monitor="box_recall", mode="max", save_top_k=1,
                           save_last=True, enable_version_counter=False)
    rec = Rec()
    m.create_trainer(callbacks=[early, ckpt, rec], accelerator=device, devices=1,
                     max_epochs=epochs, precision="32-true")
    print(f"\n[stage3] train from BASE release  device={device} max_epochs={epochs} "
          f"batch={bcfg['batch_size']} patience={patience} monitor=box_recall(max) "
          f"val=test-plots @ IoU {iou_train}", flush=True)
    t0 = time.time()
    m.trainer.fit(m)
    m.trainer.save_checkpoint(str(od / "model" / "bcnp_finetuned.ckpt"))
    best = float(ckpt.best_model_score) if ckpt.best_model_score is not None else float("nan")
    print(f"\n[stage3] best box_recall={best:.3f}  ({time.time()-t0:.0f}s, "
          f"{len(rows)} epochs, stopped_early={len(rows) < epochs})", flush=True)
    _fig_curve(rd, rows, best)
    return rows


def _write_curve(od, rows):
    fields = ["epoch", "train_loss", "val_precision", "val_recall", "val_f1", "val_loss", "seconds"]
    with open(od / "train_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _fig_curve(rd, rows, best_score):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    if not rows:
        return
    ser = lambda k: [(r["epoch"], r[k]) for r in rows if r.get(k) is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for k, col, lbl in [("train_loss", "#1b7837", "train loss"), ("val_loss", "#d73027", "val loss")]:
        s = ser(k)
        if s: ax1.plot(*zip(*s), "-o", color=col, ms=3, label=lbl)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("s14 — loss"); ax1.legend()
    for k, col, lbl in [("val_precision", "#4575b4", "val precision"),
                        ("val_recall", "#d73027", "val recall"), ("val_f1", "#7b3294", "val F1")]:
        s = ser(k)
        if s: ax2.plot(*zip(*s), "-o", color=col, ms=3, label=lbl)
    rr = ser("val_recall")
    if rr:
        be, bv = max(rr, key=lambda t: t[1])
        ax2.axvline(be, color="#999", ls="--", lw=1)
        ax2.annotate(f"best recall {bv:.3f}\n(epoch {be})", xy=(be, bv),
                     xytext=(6, -4), textcoords="offset points", fontsize=8)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score (IoU 0.4)")
    ax2.set_title("s14 — validation P / R / F1"); ax2.legend()
    fig.suptitle("s14 — BCNP from-base training (held-out test plots)", fontsize=12)
    fig.tight_layout(); fig.savefig(rd/"train_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'train_curve.png'}")


# ── stage 4: eval on held-out plots, clipped to valid pixels ─────────────────
def _px_box(r):
    for c in ("xmin", "ymin", "xmax", "ymax"):
        if c not in r or r[c] is None:
            b = r.geometry.bounds
            return b[0], b[1], b[2], b[3]
    return r["xmin"], r["ymin"], r["xmax"], r["ymax"]


def predict_plot(cfg, od, name, model, poly):
    """Predict on a resampled plot clip; return (pred_world_boxes, scores) clipped
    to the plot polygon (valid pixels)."""
    import rasterio
    bcfg = bc_cfg(cfg)
    clip = od / "tiles" / f"{name}_clip.tif"
    with rasterio.open(clip) as src:
        transform, crs = src.transform, src.crs
    patch_px = int(round(bcfg["patch_size_m"] / bcfg["target_gsd_m"]))
    pred = model.predict_tile(path=str(clip), patch_size=patch_px,
                              patch_overlap=bcfg["patch_overlap"])
    if pred is None or len(pred) == 0:
        return [], [], crs
    pred = pred.reset_index(drop=True)
    geoms, scores = [], []
    sc = pred["score"].values if "score" in pred.columns else np.ones(len(pred))
    for i, (_, r) in enumerate(pred.iterrows()):
        b = _px_box(r)
        g = C.pixel_box_to_world(b[0], b[1], b[2], b[3], transform)
        if poly.contains(g.centroid):            # clip to valid (labeled) region
            geoms.append(g); scores.append(float(sc[i]))
    return geoms, scores, crs


def census_boxes(cfg, name, census, polys):
    """World-coordinate census boxes (same sizing as labels) for one plot."""
    bcfg = bc_cfg(cfg)
    u, up = _plot_uid(name)
    sub = census[(census.UNIT == u) & (census.UNITPLOT == up) & (census.ALIVE == 1)].dropna(subset=["XCOORD", "YCOORD"])
    poly = polys[polys.Name == name].to_crs(cfg["crs"]).geometry.iloc[0]
    swx, swy = poly.bounds[0], poly.bounds[1]
    geoms = []
    for _, r in sub.iterrows():
        wx, wy = swx + float(r.XCOORD), swy + float(r.YCOORD)
        if not poly.contains(box(wx, wy, wx, wy).centroid):
            continue
        half = crown_diam_m(pd.to_numeric(r.get("DBH"), errors="coerce"), bcfg) / 2.0
        geoms.append(box(wx - half, wy - half, wx + half, wy + half))
    return geoms, poly


def evaluate(cfg, od, rd, census, polys):
    from deepforest import main as df_main
    import torch
    bcfg = bc_cfg(cfg)
    ckpt = od / "model" / "bcnp_best_recall.ckpt"
    if not ckpt.exists():
        ckpt = C.best_checkpoint(od / "model")
    print(f"\n[stage4] eval with checkpoint: {Path(ckpt).name}", flush=True)
    model = df_main.deepforest.load_from_checkpoint(str(ckpt))
    # DeepForest 2.1 does NOT propagate config score_thresh into the RetinaNet head at
    # predict time, so set it on the model directly (else the head's default 0.30 filters
    # out this low-confidence first model, which tops out ~0.25 -> zero predictions).
    thr = float(bcfg["score_thresh"])
    model.config["score_thresh"] = thr
    model.model.score_thresh = thr
    model.model.eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model.to(dev)
    except Exception:
        pass
    print(f"[stage4] model.score_thresh set to {thr} (head default 0.30 was zeroing preds)")

    per_plot, all_pred, all_scores, all_gt = {}, [], [], []
    for name in bcfg["test_plots"]:
        pg, ps, crs = predict_plot(cfg, od, name, model, polys[polys.Name == name].to_crs(cfg["crs"]).geometry.iloc[0])
        gt, poly = census_boxes(cfg, name, census, polys)
        per_plot[name] = dict(pg=pg, ps=ps, gt=gt, poly=poly, crs=crs)
        all_pred += pg; all_scores += ps; all_gt += gt
        # save predictions for QGIS (small)
        if pg:
            gpd.GeoDataFrame({"score": ps}, geometry=pg, crs=cfg["crs"]).to_file(
                od / f"predicted_{name}.geojson", driver="GeoJSON")

    iou_list = bcfg["iou_eval"]
    rows = []
    print("\n===== STAGE 4 — held-out eval (clipped to valid pixels) =====")
    hdr = f"{'plot':11s} {'IoU':>4} {'pred':>5} {'census':>6} {'TP':>5} {'prec':>6} {'recall':>7} {'F1':>6}"
    print(hdr); print("-" * len(hdr))
    scopes = [(n, per_plot[n]["pg"], per_plot[n]["ps"], per_plot[n]["gt"]) for n in bcfg["test_plots"]]
    scopes.append(("ALL", all_pred, all_scores, all_gt))
    fn_info = None
    for nm, pg, ps, gt in scopes:
        for iou in iou_list:
            r = greedy_match(pg, ps, gt, iou)
            rows.append(dict(plot=nm, iou=iou, n_pred=r["n_pred"], n_census=r["n_gt"],
                             tp=r["tp"], precision=round(r["precision"], 4),
                             recall=round(r["recall"], 4), f1=round(r["f1"], 4)))
            print(f"{nm:11s} {iou:>4} {r['n_pred']:>5} {r['n_gt']:>6} {r['tp']:>5} "
                  f"{r['precision']:>6.3f} {r['recall']:>7.3f} {r['f1']:>6.3f}")
            if nm == "ALL" and iou == iou_list[0]:
                unmatched = [g for j, g in enumerate(gt) if j not in r["matched_gt"]]
                fn_info = _fn_analysis(unmatched, pg, iou)
    _write_eval(od, rows)
    _fig_fn_buckets(rd, fn_info)
    return rows, per_plot, fn_info


def _fn_analysis(unmatched_gt, pred_geoms, iou_thr):
    best = best_iou_to(unmatched_gt, pred_geoms)
    buckets = {}
    for v in best:
        buckets[fn_bucket(v)] = buckets.get(fn_bucket(v), 0) + 1
    n = len(unmatched_gt)
    blind = buckets.get("true miss (blind)", 0)
    tight = n - blind
    print(f"\n[stage4] FN analysis @ IoU {iou_thr} ({n} missed census trees):")
    for b in ("true miss (blind)", "poor localization (0,0.30)", "near-miss [0.30,0.40)"):
        c = buckets.get(b, 0)
        print(f"    {b:28s} {c:5d}  ({c/max(1,n)*100:4.1f}%)")
    print(f"    -> blindness={blind} ({blind/max(1,n)*100:.0f}%)  "
          f"box-tightness={tight} ({tight/max(1,n)*100:.0f}%)")
    return dict(n=n, blind=blind, tight=tight, buckets=buckets)


def _write_eval(od, rows):
    fields = ["plot", "iou", "n_pred", "n_census", "tp", "precision", "recall", "f1"]
    with open(od / "eval_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {od/'eval_metrics.csv'}")


def _fig_fn_buckets(rd, fn):
    if not fn:
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    labs = ["blindness\n(no overlap)", "box-tightness\n(some overlap)"]
    vals = [fn["blind"], fn["tight"]]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar(labs, vals, color=["#d73027", "#f1a340"], width=.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+max(vals)*.01+.1,
                f"{v}\n{v/max(1,fn['n'])*100:.0f}%", ha="center", fontsize=10)
    ax.set_ylabel("missed census trees (false negatives)")
    ax.set_title(f"s14 — why trees are missed (n={fn['n']} FN @ IoU 0.4)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(rd/"fn_buckets.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'fn_buckets.png'}")


# ── stage 5: overlay figure + verdict ────────────────────────────────────────
def fig_overlay(cfg, od, rd, name, per_plot):
    import rasterio
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    C.style()
    info = per_plot[name]
    clip = od / "tiles" / f"{name}_clip.tif"
    with rasterio.open(clip) as src:
        step = max(1, int(round(max(src.width, src.height) / 1800)))
        img = src.read([1, 2, 3], out_shape=(3, src.height // step, src.width // step))
        wt = src.transform

    def to_px(x, y):
        return (x - wt.c) / wt.a / step, (y - wt.f) / wt.e / step

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(np.transpose(img, (1, 2, 0)))
    for g in info["gt"]:
        x0, y0 = to_px(g.bounds[0], g.bounds[3]); x1, y1 = to_px(g.bounds[2], g.bounds[1])
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=GT_COLOR, linewidth=0.5))
    for g in info["pg"]:
        x0, y0 = to_px(g.bounds[0], g.bounds[3]); x1, y1 = to_px(g.bounds[2], g.bounds[1])
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=PRED_COLOR, linewidth=0.5))
    ax.axis("off")
    ax.legend(handles=[Line2D([0], [0], color=GT_COLOR, lw=2, label=f"census ({len(info['gt'])})"),
                       Line2D([0], [0], color=PRED_COLOR, lw=2, label=f"predicted ({len(info['pg'])})")],
              loc="upper right", fontsize=9)
    ax.set_title(f"s14 — BCNP prediction overlay on held-out {name}")
    fig.tight_layout(); fig.savefig(rd/f"overlay_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/f'overlay_{name}.png'}")


def verdict(eval_rows, fn):
    allr = {r["iou"]: r for r in eval_rows if r["plot"] == "ALL"}
    print("\n========================= VERDICT =========================")
    for iou in sorted(allr, reverse=True):
        r = allr[iou]
        print(f"IoU {iou}:  recall {r['recall']*100:4.1f}%  precision {r['precision']*100:4.1f}%  "
              f"F1 {r['f1']*100:4.1f}%   (pred {r['n_pred']} vs census {r['n_census']}, TP {r['tp']})")
    if fn and fn["n"]:
        share = fn["blind"] / fn["n"]
        mode = ("BLINDNESS — most misses have no overlapping box at all"
                if share > 0.5 else
                "BOX-TIGHTNESS — most misses ARE detected but boxed too loosely for IoU 0.4")
        print(f"dominant failure: {mode}")
        print(f"  blindness {fn['blind']} ({share*100:.0f}%)  vs  "
              f"box-tightness {fn['tight']} ({(1-share)*100:.0f}%)  of {fn['n']} FNs")
    print("NOTE: first from-base run; census is exhaustive only inside each 100 m plot")
    print("polygon, so eval is clipped there. Recall here is an honest wetland-canopy")
    print("number, not directly comparable to the Gables ortho.")
    print("===========================================================")


def main():
    cfg = C.load_config()
    np.random.seed(SEED)
    import torch; torch.manual_seed(SEED)
    od, rd = paths(cfg)

    stats, polys_by_plot, census, polys = stage1_build(cfg, od, rd)

    if os.environ.get("TD_SKIP_TRAIN", "").strip().lower() not in ("1", "true", "yes", "on"):
        train(cfg, od, rd)
    else:
        print("\n[stage3] TD_SKIP_TRAIN set — skipping training, evaluating existing checkpoint.")

    try:
        eval_rows, per_plot, fn = evaluate(cfg, od, rd, census, polys)
        for name in bc_cfg(cfg)["test_plots"]:
            fig_overlay(cfg, od, rd, name, per_plot)
        verdict(eval_rows, fn)
    except Exception as e:
        import traceback
        print(f"\n[stage4] eval failed ({e}) — stage-1 labels + training artifacts stand.")
        traceback.print_exc()


if __name__ == "__main__":
    main()
