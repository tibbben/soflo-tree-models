"""
Step 20 — Gables FROM-SCRATCH point detector (capstone benchmark).  [local, frugal]

A third detector for the same site, architecturally MINE: not DeepForest, not YOLO,
not a wrapper around either. A compact convolutional encoder written by hand in
PyTorch plus a U-Net-style decoder that upsamples back to a 1-channel tree-CENTRE
heatmap. Ground truth is the point-first botanical inventory, so supervision is a
Gaussian rendered at each tree point and the loss is the CenterNet penalty-reduced
focal loss (weighted MSE collapses to a near-zero map — the s15 lesson). Inference is
sigmoid heatmap -> local-peak NMS by minimum distance -> tree points.

Everything downstream is deliberately IDENTICAL to how DeepForest and YOLO26s were
scored so the three numbers mean the same thing:
  - same spatial train/test AOI split (s02 clips: outputs/tiles/{train,test}_clip.tif)
  - same clean-eval region polygon (s11: outputs/clean_eval/region.geojson)
  - same distance-matched protocol as s19: peaks -> nearest inventory point, greedy
    top-confidence, one-to-one via KD-tree, best-F1 over a confidence sweep,
    P/R/F1 at 1 / 2 / 5 m, plus a random-scatter floor and edge-over-random

  stage 1  index point-supervised chips over the TRAIN clip (320x320 @ 50% overlap),
           spatial val strip held out
  stage 2  train the hand-built heatmap CNN (flips + 90 rotations + colour jitter,
           per-epoch checkpoints, early stopping on val F1)
  stage 3  inference over the whole TEST AOI -> raw local maxima -> peaks geojson
  stage 4  s19-identical distance eval (clean-eval region + full test AOI)
  stage 5  figures + the 3-way head-to-head benchmark

FRUGAL for an 18 GB Mac: rasterio WINDOWED reads only (never a 540 MB clip in RAM),
32-bit, batch 8, num_workers 0, device=mps. The clips are uncompressed and 1-row
striped, so a random 320x320 window costs ~1.7 ms — chips are therefore INDEXED
(row0/col0 in a manifest) and read on demand rather than cached as PNGs, which keeps
the disk footprint at zero. Peak RAM is printed at the end.

Writes only under outputs/gables_scratch + reports/gables_scratch.

Run:  .venv/bin/python src/s20_gables_scratch.py
      TD_SKIP_CHIPS=1 ...   # reuse cached chips
      TD_SKIP_TRAIN=1 ...   # evaluate an existing checkpoint only
      TD_EPOCHS=3 ...       # short smoke run
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
import json
import time
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box as shp_box
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEED = 0
RAW_FLOOR = 0.02        # keep local maxima above this; every threshold reuses the pass


def gs(cfg):
    return cfg["gables_scratch"]


def paths(cfg):
    od = C.p(cfg, cfg["outputs_dir"]) / "gables_scratch"
    rd = C.p(cfg, cfg["reports_dir"]) / "gables_scratch"
    (od / "chips").mkdir(parents=True, exist_ok=True)
    (od / "model").mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True, exist_ok=True)
    return od, rd


def peak_ram_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def env_on(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def inventory_points(cfg, which):
    """Point-first labels: the inventory tree points for one AOI, as (N,2) world xy.

    Uses the SAME files DeepForest was trained/scored on (s02 outputs). Centroids of
    the label boxes are the inventory points (s01 builds one box per confirmed tree),
    which is exactly what s19 matches against.
    """
    g = gpd.read_file(C.p(cfg, cfg["outputs_dir"]) / f"boxes_{which}.geojson").to_crs(cfg["crs"])
    c = g.geometry.centroid
    return np.stack([c.x.values, c.y.values], 1)


# ── stage 1: point-supervised chips (lazy windowed reads) ───────────────────
def build_chips(cfg, od, rd):
    import rasterio
    from rasterio.transform import rowcol
    cf = gs(cfg)
    S = int(cf["tile_px"]); stride = int(round(S * (1 - cf["overlap"])))
    min_valid = float(cf["min_valid_frac"]); min_trees = int(cf["min_trees_per_chip"])
    cap = int(cf["max_train_chips"]); bg_ratio = int(cf["bg_ratio"])
    rng = np.random.default_rng(SEED)

    clip = C.p(cfg, cfg["outputs_dir"]) / "tiles" / "train_clip.tif"
    pts = inventory_points(cfg, "train")
    cdir = od / "chips"
    print("[stage1] indexing point-supervised chips over the TRAIN AOI clip ...", flush=True)
    print(f"         {S}px @ {int(cf['overlap']*100)}% overlap (stride {stride}px), "
          f"sigma {cf['sigma_px']}px, cap {cap} train chips, bg 1-per-{bg_ratio}", flush=True)

    with rasterio.open(clip) as src:
        tf = src.transform; gsd = abs(tf.a); H, W = src.height, src.width
        rows_c, cols_c = rowcol(tf, pts[:, 0], pts[:, 1])
        px_x = np.asarray(cols_c, float); px_y = np.asarray(rows_c, float)
        inside = (px_x >= 0) & (px_x < W) & (px_y >= 0) & (px_y < H)
        print(f"         clip {W}x{H} px @ {gsd*100:.1f} cm | inventory points "
              f"{len(pts)} ({int(inside.sum())} inside the clip)", flush=True)

        # spatial val strip: the TOP rows of the train AOI holding val_frac_trees of the
        # trees, with a one-chip buffer so no pixel is ever shared between train and val.
        vrow = int(np.percentile(px_y[inside], float(cf["val_frac_trees"]) * 100))
        tree_cand, bg_cand = [], []
        for row0 in range(0, H - S + 1, stride):
            if vrow <= row0 < vrow + S:          # buffer band — belongs to neither
                continue
            split = "val" if row0 + S <= vrow else "train"
            for col0 in range(0, W - S + 1, stride):
                m = ((px_x >= col0) & (px_x < col0 + S) &
                     (px_y >= row0) & (px_y < row0 + S) & inside)
                (tree_cand if m.sum() >= min_trees else bg_cand).append((row0, col0, m, split))

        # frugality caps on the tree chips (val is only the early-stopping signal)
        def subsample(items, k):
            if len(items) <= k:
                return items
            return [items[i] for i in sorted(rng.choice(len(items), k, replace=False))]
        tr_tree = subsample([c for c in tree_cand if c[3] == "train"], cap)
        va_tree = subsample([c for c in tree_cand if c[3] == "val"], int(cf["max_val_chips"]))
        # background negatives (no tree at all) teach the model what is NOT a crown
        def take_bg(split, n_tree):
            pool = [c for c in bg_cand if c[3] == split]
            k = min(len(pool), max(0, n_tree // bg_ratio))
            if k == 0:
                return []
            return [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]
        cand = tr_tree + va_tree + take_bg("train", len(tr_tree)) + take_bg("val", len(va_tree))

        manifest, point_rows = [], []
        stats = {"train": dict(chips=0, trees=0, bg=0), "val": dict(chips=0, trees=0, bg=0)}
        for row0, col0, m, split in cand:
            arr = src.read((1, 2, 3), window=((row0, row0 + S), (col0, col0 + S)),
                           boundless=True, fill_value=0)
            if (arr.max(0) > 8).mean() < min_valid:      # mostly nodata — not trainable
                continue
            name = f"r{row0}_c{col0}"
            xs = px_x[m] - col0; ys = px_y[m] - row0
            for xx, yy in zip(xs, ys):
                point_rows.append(dict(chip=name, x=round(float(xx), 1), y=round(float(yy), 1)))
            n = int(m.sum())
            manifest.append(dict(chip=name, split=split, row0=row0, col0=col0, n_trees=n))
            stats[split]["chips"] += 1; stats[split]["trees"] += n
            stats[split]["bg"] += int(n == 0)

    pd.DataFrame(manifest).to_csv(cdir / "manifest.csv", index=False)
    pd.DataFrame(point_rows).to_csv(cdir / "points.csv", index=False)
    (cdir / "meta.json").write_text(json.dumps(dict(gsd=gsd, tile_px=S, clip=str(clip))))

    print("\n===== STAGE 1 — chips =====")
    for split in ("train", "val"):
        s = stats[split]
        print(f"  {split:5s}: chips={s['chips']:5d}  labelled trees={s['trees']:6d}  "
              f"background chips={s['bg']:4d}")
    print(f"  val strip = the top rows of the TRAIN AOI holding {cf['val_frac_trees']*100:.0f}% "
          f"of its trees (spatially disjoint, one-chip buffer).")
    print("  The TEST AOI is never seen in training.")
    rows = [dict(split=k, **v) for k, v in stats.items()]
    with open(od / "chip_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "chips", "trees", "bg"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {od/'chip_counts.csv'}")
    return stats


# ── Gaussian heatmap target ─────────────────────────────────────────────────
def render_heatmap(points_xy, S, sigma):
    """Max-merged 2D Gaussians (peak = 1.0 at each tree point) on an SxS float32 map."""
    hm = np.zeros((S, S), np.float32)
    if len(points_xy) == 0:
        return hm
    r = int(round(3 * sigma))
    ay = np.arange(-r, r + 1)[:, None]; ax = np.arange(-r, r + 1)[None, :]
    g = np.exp(-(ax * ax + ay * ay) / (2 * sigma * sigma)).astype(np.float32)
    for x, y in points_xy:
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(S, xi + r + 1)
        y0, y1 = max(0, yi - r), min(S, yi + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        gx0, gy0 = x0 - (xi - r), y0 - (yi - r)
        patch = g[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]
        np.maximum(hm[y0:y1, x0:x1], patch, out=hm[y0:y1, x0:x1])
    return hm


# ── dataset ─────────────────────────────────────────────────────────────────
try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    _TorchDataset = object


class GablesChipDS(_TorchDataset):
    """Indexed chips -> (normalised RGB tensor, Gaussian heatmap tensor).

    The imagery is read straight out of the train clip as a 320x320 window at access
    time (~1.7 ms; the clip is uncompressed and 1-row striped) rather than cached as
    PNGs — no disk footprint, and never more than one chip in memory.

    Aerial augmentation: 90 degree rotations, horizontal/vertical flips (points
    transformed with the image), brightness/contrast/per-channel colour jitter.
    """

    def __init__(self, od, split, sigma, augment, tile_px, clip, out_stride=4):
        cdir = Path(od) / "chips"
        man = pd.read_csv(cdir / "manifest.csv")
        man = man[man.split == split].reset_index(drop=True)
        pts = pd.read_csv(cdir / "points.csv") if (cdir / "points.csv").exists() else \
            pd.DataFrame(columns=["chip", "x", "y"])
        self.by_chip = {c: g[["x", "y"]].values.astype(np.float32) for c, g in pts.groupby("chip")}
        self.rows = man.to_dict("records")
        self.sigma = float(sigma); self.augment = bool(augment)
        self.S = int(tile_px); self.clip = str(clip); self._src = None
        self.stride = int(out_stride)

    def __len__(self):
        return len(self.rows)

    def points(self, i):
        return self.by_chip.get(self.rows[i]["chip"], np.zeros((0, 2), np.float32))

    def read_rgb(self, i):
        import rasterio
        if self._src is None:                     # opened lazily, one handle per process
            self._src = rasterio.open(self.clip)
        r = self.rows[i]; S = self.S
        arr = self._src.read((1, 2, 3), window=((r["row0"], r["row0"] + S),
                                                (r["col0"], r["col0"] + S)),
                             boundless=True, fill_value=0)
        rgb = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
        return rgb * (rgb.max(2) > 8)[..., None]  # zero the nodata margin

    def __getitem__(self, i):
        import torch
        chip = self.rows[i]["chip"]
        img = self.read_rgb(i)
        S = img.shape[0]
        p = self.by_chip.get(chip, np.zeros((0, 2), np.float32)).copy()
        if self.augment:
            rng = np.random.default_rng((SEED + i * 7919 + int(time.time() * 1e3)) % (2**32))
            k = int(rng.integers(0, 4))
            for _ in range(k):
                img = np.rot90(img).copy()
                if len(p):
                    p = np.stack([p[:, 1], (S - 1) - p[:, 0]], 1)
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
                if len(p): p[:, 0] = (S - 1) - p[:, 0]
            if rng.random() < 0.5:
                img = img[::-1].copy()
                if len(p): p[:, 1] = (S - 1) - p[:, 1]
            imgf = img.astype(np.float32) / 255.0
            imgf *= (1 + (rng.random() - 0.5) * 0.4)                          # brightness
            mean = imgf.mean()
            imgf = mean + (imgf - mean) * (1 + (rng.random() - 0.5) * 0.4)    # contrast
            imgf *= (1 + (rng.random(3).astype(np.float32) - 0.5) * 0.16)     # colour cast
            imgf = np.clip(imgf, 0, 1)
        else:
            imgf = img.astype(np.float32) / 255.0
        # target lives at the head's resolution: points and sigma scale by the stride
        hm = render_heatmap(p / self.stride, S // self.stride, self.sigma / self.stride)
        x = (imgf - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.transpose(x, (2, 0, 1)).copy())
        return x, torch.from_numpy(hm[None])


# ── the model: a hand-written encoder-decoder heatmap CNN ───────────────────
def build_model(width=24, out_stride=4):
    """TreeNet — a compact custom CNN, written here rather than imported.

    No torchvision backbone, no pretrained weights, no detection library: a stem plus
    four stride-2 residual encoder stages (/1 -> /16) and bilinear-upsample decoder
    stages with skip connections back up to `out_stride`, ending in a 1x1 head that
    emits a single tree-centre confidence channel. ~2 M parameters; the /16 bottleneck
    sees roughly a 10 m context window at 5 cm/px, about one and a half campus crowns.

    The head sits at 1/4 resolution by default (the CenterNet convention): decoding all
    the way to 1/1 multiplies the negative pixels per positive by 16 and the focal loss
    responds by pinning every peak near 0.02 instead of learning confident centres.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def cbr(i, o, k=3, s=1):
        return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class ResBlock(nn.Module):
        """Two 3x3 convs with an identity shortcut (projected when widths differ)."""

        def __init__(self, i, o):
            super().__init__()
            self.c1 = cbr(i, o)
            self.c2 = nn.Sequential(nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
            self.skip = None if i == o else nn.Sequential(
                nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o))
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            idt = x if self.skip is None else self.skip(x)
            return self.act(self.c2(self.c1(x)) + idt)

    class Down(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.body = nn.Sequential(cbr(i, o, s=2), ResBlock(o, o))

        def forward(self, x):
            return self.body(x)

    class Up(nn.Module):
        def __init__(self, i, skip_ch, o):
            super().__init__()
            self.body = ResBlock(i + skip_ch, o)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.body(torch.cat([x, skip], 1))

    class TreeNet(nn.Module):
        def __init__(self, w, stride):
            super().__init__()
            self.stride = stride
            c1, c2, c3, c4, c5 = w, w * 2, w * 4, w * 6, w * 8
            self.stem = nn.Sequential(cbr(3, c1), ResBlock(c1, c1))   # /1
            self.d1 = Down(c1, c2)                                    # /2
            self.d2 = Down(c2, c3)                                    # /4
            self.d3 = Down(c3, c4)                                    # /8
            self.d4 = Down(c4, c5)                                    # /16
            self.u4 = Up(c5, c4, c4)                                  # /8
            self.u3 = Up(c4, c3, c3)                                  # /4
            self.u2 = Up(c3, c2, c2) if stride <= 2 else None         # /2
            self.u1 = Up(c2, c1, c1) if stride <= 1 else None         # /1
            out_ch = c3 if stride == 4 else (c2 if stride == 2 else c1)
            self.head = nn.Conv2d(out_ch, 1, 1)
            # CenterNet-style prior: start near p=0.1 so focal loss does not saturate
            nn.init.constant_(self.head.bias, -2.19)
            nn.init.normal_(self.head.weight, std=0.01)

        def forward(self, x):
            x1 = self.stem(x)
            x2 = self.d1(x1)
            x3 = self.d2(x2)
            x4 = self.d3(x3)
            x5 = self.d4(x4)
            d = self.u4(x5, x4)
            d = self.u3(d, x3)
            if self.u2 is not None:
                d = self.u2(d, x2)
            if self.u1 is not None:
                d = self.u1(d, x1)
            return torch.sigmoid(self.head(d))

    return TreeNet(int(width), int(out_stride))


def focal_heatmap_loss(pred, target, alpha=2.0, beta=4.0, eps=1e-6):
    """CenterNet penalty-reduced focal loss. Positives are the exact peak pixels
    (target == 1); every other pixel is a negative whose penalty is down-weighted by
    (1-target)^beta so the Gaussian skirt around a tree is forgiven. Normalised by the
    number of positives — robust to the extreme fg/bg imbalance that collapses weighted
    MSE to a near-zero map (measured in s15)."""
    import torch
    p = pred.clamp(eps, 1 - eps)
    pos = target.ge(1.0 - 1e-4).float()
    neg = 1.0 - pos
    neg_w = (1.0 - target).clamp(min=0).pow(beta)
    pos_loss = (1 - p).pow(alpha) * torch.log(p) * pos
    neg_loss = p.pow(alpha) * torch.log(1 - p) * neg_w * neg
    npos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / npos


# ── peak finding + matching ─────────────────────────────────────────────────
def find_peaks(hm, thr, min_dist_px):
    from scipy.ndimage import maximum_filter
    k = max(3, int(min_dist_px) | 1)
    mx = maximum_filter(hm, size=k, mode="constant")
    pk = (hm == mx) & (hm >= thr)
    ys, xs = np.where(pk)
    return xs, ys, hm[ys, xs]


def nms_points(pts_xy, scores, min_dist):
    """Greedy NMS: keep the highest-scoring peak, drop others within min_dist."""
    if len(pts_xy) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    from scipy.spatial import cKDTree
    order = np.argsort(-np.asarray(scores))
    pts = np.asarray(pts_xy)[order]; sc = np.asarray(scores)[order]
    keep = np.ones(len(pts), bool)
    tree = cKDTree(pts)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(pts[i], min_dist):
            if j > i and keep[j]:
                keep[j] = False
    return pts[keep], sc[keep]


def match_distance(pred_xy, pred_sc, gt_xy, radius):
    """s19-identical: nearest inventory point, greedy top-confidence, one-to-one."""
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


# ── stage 2: train ──────────────────────────────────────────────────────────
def conf_sweep(scores, n):
    """Thresholds at quantiles of the model's own peak-score distribution, so the
    sweep always resolves the useful range whatever scale the head settles on."""
    if len(scores) == 0:
        return [RAW_FLOOR]
    qs = np.quantile(np.asarray(scores, float), np.linspace(0.0, 0.995, int(n)))
    return sorted({round(float(q), 5) for q in qs})


def val_f1(model, ds, device, cfg, batch=8):
    """Best-F1 over the confidence sweep on the held-out val strip, matched in chip
    pixel space at val_match_radius_m. This is the early-stopping signal: the loss
    number is not the thing we care about, the detection F1 is."""
    import torch
    cf = gs(cfg); gsd = 0.05; stride = int(cf["out_stride"])
    # everything below is in HEAD pixels (1/stride of the chip)
    min_dist_px = float(cf["peak_min_dist_m"]) / gsd / stride
    rad_px = float(cf["val_match_radius_m"]) / gsd / stride
    per_chip, all_sc = [], []
    model.eval()
    with torch.no_grad():
        for i0 in range(0, len(ds), batch):
            xs = [ds[i][0] for i in range(i0, min(i0 + batch, len(ds)))]
            x = torch.stack(xs).to(device)
            hms = model(x).cpu().numpy()[:, 0]
            for k, hm in enumerate(hms):
                px, py, sc = find_peaks(hm, RAW_FLOOR, min_dist_px)
                per_chip.append((np.stack([px, py], 1).astype(float), sc,
                                 np.asarray(ds.points(i0 + k), float) / stride))
                all_sc.append(sc)
    sweep = conf_sweep(np.concatenate(all_sc) if all_sc else np.zeros(0),
                       cf["conf_sweep_points"])
    best = dict(f1=0.0, conf=float(sweep[0]), precision=0.0, recall=0.0)
    for thr in sweep:
        P, Sc, G = [], [], []
        off = 0.0
        for pts, sc, gt in per_chip:
            m = sc >= thr
            p, s = nms_points(pts[m], sc[m], min_dist_px)
            # offset each chip into its own slab so cross-chip matches are impossible
            if len(p):
                P.append(p + off); Sc.append(s)
            if len(gt):
                G.append(np.asarray(gt, float) + off)
            off += 10000.0
        pxy = np.vstack(P) if P else np.zeros((0, 2))
        psc = np.concatenate(Sc) if Sc else np.zeros(0)
        gxy = np.vstack(G) if G else np.zeros((0, 2))
        r = match_distance(pxy, psc, gxy, rad_px)
        if r["f1"] > best["f1"]:
            best = dict(f1=r["f1"], conf=float(thr), precision=r["precision"], recall=r["recall"])
    return best


def train(cfg, od, rd):
    import torch
    from torch.utils.data import DataLoader
    cf = gs(cfg)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sigma = float(cf["sigma_px"])
    epochs = int(os.environ.get("TD_EPOCHS", cf["epochs"]))
    patience = int(cf["early_stop_patience"])
    bs = int(cf["batch_size"]); nw = int(cf["num_workers"])

    clip = C.p(cfg, cfg["outputs_dir"]) / "tiles" / "train_clip.tif"
    ost = int(cf["out_stride"])
    tr = GablesChipDS(od, "train", sigma, True, cf["tile_px"], clip, ost)
    va = GablesChipDS(od, "val", sigma, False, cf["tile_px"], clip, ost)
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)

    model = build_model(cf["width"], ost).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"\n[stage2] train TreeNet (hand-built, {nparam/1e6:.2f} M params, width={cf['width']}) "
          f"device={device} epochs={epochs} batch={bs} lr={cf['lr']} loss=focal", flush=True)
    print(f"         train chips={len(tr)}  val chips={len(va)}  "
          f"early stop on val F1 @ {cf['val_match_radius_m']} m (patience {patience})", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=float(cf["lr"]),
                           weight_decay=float(cf["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    rows, best_f1, bad = [], -1.0, 0
    for ep in range(epochs):
        t0 = time.time(); model.train(); tot = 0.0; n = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = focal_heatmap_loss(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        tl = tot / max(1, n)
        vb = val_f1(model, va, device, cfg, batch=bs)
        sched.step(vb["f1"])
        secs = time.time() - t0
        rows.append(dict(epoch=ep, train_loss=round(tl, 6), val_f1=round(vb["f1"], 4),
                         val_precision=round(vb["precision"], 4), val_recall=round(vb["recall"], 4),
                         val_best_conf=vb["conf"], lr=opt.param_groups[0]["lr"],
                         seconds=round(secs, 1)))
        _write_curve(od, rows)
        torch.save(model.state_dict(), od / "model" / f"epoch_{ep:03d}.pt")
        _prune_ckpts(od, int(cf.get("keep_last_ckpts", 3)))
        print(f"   [epoch {ep:02d}] {secs:6.1f}s loss={tl:.4f} val_F1={vb['f1']:.4f} "
              f"(P {vb['precision']:.3f} R {vb['recall']:.3f} @conf {vb['conf']}) "
              f"peakRAM={peak_ram_gb():.1f}GB", flush=True)
        if vb["f1"] > best_f1 + 1e-5:
            best_f1, bad = vb["f1"], 0
            torch.save(model.state_dict(), od / "model" / "scratch_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("   early stop (no val-F1 improvement)"); break
    torch.save(model.state_dict(), od / "model" / "scratch_last.pt")
    print(f"[stage2] best val F1 = {best_f1:.4f} over {len(rows)} epochs, "
          f"peakRAM={peak_ram_gb():.1f}GB", flush=True)
    _fig_curve(rd, rows)
    return rows


def _prune_ckpts(od, keep):
    """A checkpoint is written every epoch, but only the most recent `keep` are
    retained (plus scratch_best.pt / scratch_last.pt). This machine's disk is nearly
    full; 60 epochs x ~8 MB would not fit."""
    ck = sorted((od / "model").glob("epoch_*.pt"))
    for old in ck[:-keep] if keep > 0 else ck:
        old.unlink(missing_ok=True)


def _write_curve(od, rows):
    fields = ["epoch", "train_loss", "val_f1", "val_precision", "val_recall",
              "val_best_conf", "lr", "seconds"]
    with open(od / "train_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def _fig_curve(rd, rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    if not rows:
        return
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(ep, [r["train_loss"] for r in rows], "-o", color="#4575b4", ms=3,
            label="train focal loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("focal loss", color="#4575b4")
    ax2 = ax.twinx()
    ax2.plot(ep, [r["val_f1"] for r in rows], "-o", color="#1b7837", ms=3, label="val F1 (2 m)")
    ax2.set_ylabel("val F1 @ 2 m", color="#1b7837"); ax2.set_ylim(0, 1)
    be = max(rows, key=lambda r: r["val_f1"])
    ax2.axvline(be["epoch"], color="#999", ls="--", lw=1)
    ax2.annotate(f"best val F1 {be['val_f1']:.3f}\n(epoch {be['epoch']})",
                 xy=(be["epoch"], be["val_f1"]), xytext=(6, -22),
                 textcoords="offset points", fontsize=8)
    for sp in ("top",):
        ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax.set_title("s20 — from-scratch heatmap CNN training (Gables)")
    fig.tight_layout(); fig.savefig(rd / "train_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {rd/'train_curve.png'}")


# ── stage 3: inference over the whole test AOI ──────────────────────────────
def predict_test(cfg, od, model, device):
    """Stream 320x320 windows across the TEST clip, collect ALL local maxima above a
    low floor in world coordinates. Peak LOCATIONS are threshold-independent, so one
    pass serves the whole confidence sweep."""
    import rasterio
    from rasterio.transform import xy
    import torch
    cf = gs(cfg)
    S = int(cf["tile_px"]); step = int(round(S * (1 - float(cf["infer_overlap"]))))
    ostride = int(cf["out_stride"])
    clip = C.p(cfg, cfg["outputs_dir"]) / "tiles" / "test_clip.tif"
    bs = int(cf["batch_size"])
    world, score = [], []
    t0 = time.time()
    model.eval()
    with rasterio.open(clip) as src:
        tf = src.transform; gsd = abs(tf.a); H, W = src.height, src.width
        min_dist_px = float(cf["peak_min_dist_m"]) / gsd / ostride     # in head pixels
        starts = [(r, c) for r in range(0, H - S + 1, step) for c in range(0, W - S + 1, step)]
        print(f"\n[stage3] inference over the TEST AOI: {len(starts)} windows "
              f"({S}px, stride {step}px) streamed from {clip.name}", flush=True)
        batch_x, batch_rc, done, skipped = [], [], 0, 0
        with torch.no_grad():
            for row0, col0 in starts:
                arr = src.read((1, 2, 3), window=((row0, row0 + S), (col0, col0 + S)),
                               boundless=True, fill_value=0)
                rgb = np.transpose(arr, (1, 2, 0)).astype(np.float32)
                valid = rgb.max(2) > 8
                if valid.mean() < 0.2:            # essentially all nodata — nothing to find
                    skipped += 1; done += 1
                    continue
                img = (rgb * valid[..., None]) / 255.0
                x = (img - IMAGENET_MEAN) / IMAGENET_STD
                batch_x.append(np.transpose(x, (2, 0, 1)))
                batch_rc.append((row0, col0))
                if len(batch_x) == bs:
                    done += len(batch_x)
                    _run_batch(model, device, batch_x, batch_rc, tf, min_dist_px,
                               ostride, world, score)
                    batch_x, batch_rc = [], []
                    if done % 400 < bs:
                        print(f"   {done}/{len(starts)} windows  ({len(world)} raw peaks, "
                              f"{time.time()-t0:.0f}s, peakRAM={peak_ram_gb():.1f}GB)", flush=True)
            if batch_x:
                _run_batch(model, device, batch_x, batch_rc, tf, min_dist_px,
                           ostride, world, score)
    xy_arr = np.array(world).reshape(-1, 2); sc = np.array(score)
    print(f"[stage3] {len(xy_arr)} raw local maxima >= {RAW_FLOOR} "
          f"({skipped} nodata windows skipped, {time.time()-t0:.0f}s)", flush=True)
    if len(sc):
        print(f"         score range {sc.min():.3f} - {sc.max():.3f}, "
              f"median {np.median(sc):.3f}", flush=True)
    np.savez_compressed(od / "test_raw_peaks.npz", xy=xy_arr, score=sc)
    return xy_arr, sc


def _run_batch(model, device, batch_x, batch_rc, tf, min_dist_px, ostride, world, score):
    import torch
    from rasterio.transform import xy
    x = torch.from_numpy(np.stack(batch_x)).to(device)
    hms = model(x).cpu().numpy()[:, 0]
    half = (ostride - 1) / 2.0            # head pixel i covers full-res [i*s, i*s+s)
    for hm, (row0, col0) in zip(hms, batch_rc):
        px, py, sc = find_peaks(hm, RAW_FLOOR, min_dist_px)
        for xi, yi, s in zip(px, py, sc):
            wx, wy = xy(tf, row0 + yi * ostride + half, col0 + xi * ostride + half)
            world.append((wx, wy)); score.append(float(s))


_PEAK_CACHE = {}


def peaks_at_conf(cfg, raw_xy, raw_sc, conf):
    """Threshold the raw local maxima, then NMS-merge across the whole AOI (this is
    where duplicates from overlapping inference windows collapse). Memoised: the
    sweep asks for the same thresholds once per radius and per scope."""
    key = round(float(conf), 6)
    if key not in _PEAK_CACHE:
        m = raw_sc >= conf
        _PEAK_CACHE[key] = nms_points(raw_xy[m], raw_sc[m], float(gs(cfg)["peak_min_dist_m"]))
    return _PEAK_CACHE[key]


# ── stage 4: s19-identical distance eval ────────────────────────────────────
def random_floor(poly, n_pred, gt_xy, radius, trials=16):
    """s19-identical: scatter the same number of points uniformly in the scope and
    match them the same way. Density plus a loose radius alone scores non-zero, so
    this is the floor the model must clear."""
    if n_pred == 0 or len(gt_xy) == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    rng = np.random.default_rng(0)
    minx, miny, maxx, maxy = poly.bounds
    Ps, Rs, F = [], [], []
    for _ in range(trials):
        xs = rng.uniform(minx, maxx, n_pred); ys = rng.uniform(miny, maxy, n_pred)
        r = match_distance(np.stack([xs, ys], 1), rng.random(n_pred), gt_xy, radius)
        Ps.append(r["precision"]); Rs.append(r["recall"]); F.append(r["f1"])
    return dict(precision=float(np.mean(Ps)), recall=float(np.mean(Rs)), f1=float(np.mean(F)))


def _in_poly(xy_arr, poly):
    if len(xy_arr) == 0:
        return np.zeros(0, bool)
    import shapely
    return np.asarray(shapely.contains_xy(poly, xy_arr[:, 0], xy_arr[:, 1]), bool)


def evaluate(cfg, od, rd, raw_xy, raw_sc):
    cf = gs(cfg)
    root = C.p(cfg, cfg["outputs_dir"])
    inv = gpd.read_file(root / "boxes_test.geojson").to_crs(cfg["crs"])
    region = gpd.read_file(root / "clean_eval" / "region.geojson").to_crs(cfg["crs"]).geometry.iloc[0]
    aoi = shp_box(*inv.total_bounds)
    ic = inv.geometry.centroid
    inv_xy = np.stack([ic.x.values, ic.y.values], 1)

    scopes = {"clean_eval_region": region, "full_test_aoi": aoi}
    sweep = conf_sweep(raw_sc, cf["conf_sweep_points"])
    rows, best_by_scope = [], {}
    print("\n===== STAGE 4 — Gables DISTANCE-matched eval (s20 from-scratch peaks) =====")
    print("(peaks -> nearest inventory point, greedy top-conf, one-to-one, KD-tree; "
          "protocol identical to s19)")
    print(f"confidence sweep: {len(sweep)} quantiles of the peak-score distribution, "
          f"{sweep[0]:.3f} - {sweep[-1]:.3f}\n")
    for sname, poly in scopes.items():
        gm = _in_poly(inv_xy, poly)
        gxy = inv_xy[gm]
        print(f"--- {sname}: {len(gxy)} inventory points ---")
        for r in cf["match_radii_m"]:
            best = dict(f1=-1)
            for thr in sweep:
                pxy, psc = peaks_at_conf(cfg, raw_xy, raw_sc, float(thr))
                pm = _in_poly(pxy, poly)
                m = match_distance(pxy[pm], psc[pm], gxy, r)
                if m["f1"] > best["f1"]:
                    best = dict(conf=float(thr), **m)
            rf = random_floor(poly, best["n_pred"], gxy, r)
            edge = best["recall"] - rf["recall"]
            rows.append(dict(scope=sname, radius_m=r, best_conf=best["conf"],
                             n_pred=best["n_pred"], n_inventory=len(gxy), tp=best["tp"],
                             precision=round(best["precision"], 4),
                             recall=round(best["recall"], 4), f1=round(best["f1"], 4),
                             rand_recall=round(rf["recall"], 4),
                             rand_precision=round(rf["precision"], 4),
                             edge_recall=round(edge, 4)))
            best_by_scope[(sname, r)] = best
            print(f"   r={r} m  best@conf {best['conf']}: P {best['precision']:.3f}  "
                  f"R {best['recall']:.3f}  F1 {best['f1']:.3f}  | random R {rf['recall']:.3f}  "
                  f"edge {edge:+.3f}  (pred {best['n_pred']}, inv {len(gxy)}, TP {best['tp']})")
        print()

    out = od / "gables_scratch_distance_eval.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")

    # committed prediction layer at the headline (5 m clean-region) operating point
    hconf = best_by_scope[("clean_eval_region", 5.0)]["conf"]
    pxy, psc = peaks_at_conf(cfg, raw_xy, raw_sc, hconf)
    gpd.GeoDataFrame({"score": np.round(psc, 4)},
                     geometry=gpd.points_from_xy(pxy[:, 0], pxy[:, 1]),
                     crs=cfg["crs"]).to_file(od / "predicted_points.geojson", driver="GeoJSON")
    print(f"wrote {od/'predicted_points.geojson'}  ({len(pxy)} peaks @ conf {hconf})")
    return rows, hconf


# ── stage 5: figures + 3-way benchmark ──────────────────────────────────────
def fig_heatmap_chip(cfg, od, rd, model, device, conf):
    import torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    cf = gs(cfg)
    # a real held-out TEST tile (never seen in training), taken from the test clip
    import rasterio
    root = C.p(cfg, cfg["outputs_dir"])
    inv_xy = inventory_points(cfg, "test")
    S = int(cf["tile_px"])
    with rasterio.open(root / "tiles" / "test_clip.tif") as src:
        from rasterio.transform import rowcol
        tf = src.transform; gsd = abs(tf.a)
        rr, cc = rowcol(tf, inv_xy[:, 0], inv_xy[:, 1])
        rr = np.asarray(rr, float); cc = np.asarray(cc, float)
        best = None
        for row0 in range(0, src.height - S + 1, S):
            for col0 in range(0, src.width - S + 1, S):
                m = ((cc >= col0) & (cc < col0 + S) & (rr >= row0) & (rr < row0 + S))
                if best is None or m.sum() > best[0]:
                    best = (int(m.sum()), row0, col0, m)
        n, row0, col0, m = best
        arr = src.read((1, 2, 3), window=((row0, row0 + S), (col0, col0 + S)))
    img = np.transpose(arr, (1, 2, 0)).astype(np.float32) / 255.0
    x = (img - IMAGENET_MEAN) / IMAGENET_STD
    xt = torch.from_numpy(np.transpose(x, (2, 0, 1))[None].copy()).to(device)
    model.eval()
    with torch.no_grad():
        hm = model(xt)[0, 0].cpu().numpy()
    ost = int(cf["out_stride"])
    px, py, _ = find_peaks(hm, conf, float(cf["peak_min_dist_m"]) / gsd / ost)
    px = px * ost + (ost - 1) / 2.0       # head pixels -> chip pixels, for the overlay
    py = py * ost + (ost - 1) / 2.0
    gx, gy = cc[m] - col0, rr[m] - row0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    axes[0].imshow(img); axes[0].set_title(f"held-out test tile ({S}px = {S*gsd:.0f} m)\n"
                                           f"inventory points (n={n})")
    axes[0].scatter(gx, gy, s=26, c="#39ff14", marker="+", linewidths=1.0)
    hmax = float(hm.max())
    vmax = float(np.clip(hmax * 1.05, 0.25, 1.0))   # faint maps stay readable, scale stated
    im = axes[1].imshow(hm, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title(f"predicted tree-centre heatmap\n"
                      f"(sigmoid, max={hmax:.2f}; colour scale 0-{vmax:.2f})")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(img)
    axes[2].scatter(gx, gy, s=26, c="#39ff14", marker="+", linewidths=1.0, label=f"inventory ({n})")
    axes[2].scatter(px, py, s=44, facecolors="none", edgecolors="#d73027", linewidths=1.3,
                    label=f"peaks @ conf {conf} ({len(px)})")
    axes[2].legend(loc="upper right", fontsize=8, markerscale=1.2)
    axes[2].set_title("local-peak NMS -> predicted trees")
    for a in axes:
        a.axis("off")
    fig.suptitle("s20 — from-scratch heatmap CNN on a held-out Gables tile", fontsize=12, y=1.04)
    fig.tight_layout(); fig.savefig(rd / "heatmap_tile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {rd/'heatmap_tile.png'}")


def fig_overlay(cfg, od, rd, raw_xy, raw_sc, conf):
    import rasterio
    from rasterio.windows import from_bounds
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    root = C.p(cfg, cfg["outputs_dir"])
    region = gpd.read_file(root / "clean_eval" / "region.geojson").to_crs(cfg["crs"]).geometry.iloc[0]
    minx, miny, maxx, maxy = region.bounds
    inv = gpd.read_file(root / "boxes_test.geojson").to_crs(cfg["crs"])
    ic = inv.geometry.centroid
    inv_xy = np.stack([ic.x.values, ic.y.values], 1)
    inv_xy = inv_xy[_in_poly(inv_xy, region)]
    pxy, psc = peaks_at_conf(cfg, raw_xy, raw_sc, conf)
    pm = _in_poly(pxy, region); pxy = pxy[pm]

    with rasterio.open(root / "tiles" / "test_clip.tif") as src:
        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        step = max(1, int(round(max(win.width, win.height) / 2000)))
        img = src.read([1, 2, 3], window=win,
                       out_shape=(3, int(win.height) // step, int(win.width) // step))
        wt = src.window_transform(win)
    H, W = img.shape[1], img.shape[2]

    def to_px(x, y):
        return (x - wt.c) / wt.a / step, (y - wt.f) / wt.e / step

    fig, ax = plt.subplots(figsize=(11.5, 11.5))
    ax.imshow(np.transpose(img, (1, 2, 0)))
    g = [to_px(x, y) for x, y in inv_xy]
    p = [to_px(x, y) for x, y in pxy]
    ax.scatter([q[0] for q in g], [q[1] for q in g], s=26, c="#39ff14", marker="+",
               linewidths=0.9, label=f"inventory ({len(g)})")
    ax.scatter([q[0] for q in p], [q[1] for q in p], s=34, facecolors="none",
               edgecolors="#d73027", linewidths=0.9, label=f"predicted peaks ({len(p)})")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.legend(loc="upper right", fontsize=10, markerscale=1.6)
    ax.set_title("s20 — from-scratch predictions vs inventory, s11 clean-eval region "
                 f"(250 x 250 m, conf {conf})")
    fig.tight_layout(); fig.savefig(rd / "predictions_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {rd/'predictions_overlay.png'}")


def benchmark(cfg, od, rd, rows):
    """The 3-way head-to-head: this from-scratch model vs DeepForest (s19) vs YOLO26s."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    cf = gs(cfg); base = cf["baselines"]
    mine = {r["radius_m"]: r for r in rows if r["scope"] == "clean_eval_region"}
    df = pd.read_csv(C.p(cfg, cfg["outputs_dir"]) / "gables_distance_eval.csv")
    df = df[df.scope == "clean_eval_region"].set_index("radius_m")

    out = []
    for r in cf["match_radii_m"]:
        m = mine[r]
        out.append(dict(model="s20_from_scratch", radius_m=r, precision=m["precision"],
                        recall=m["recall"], f1=m["f1"]))
        if r in df.index:
            d = df.loc[r]
            out.append(dict(model="deepforest_s19", radius_m=r, precision=round(float(d.precision), 4),
                            recall=round(float(d.recall), 4), f1=round(float(d.f1), 4)))
    out.append(dict(model="yolo26s_ahsan", radius_m=5.0, precision=base["yolo26s"]["precision"],
                    recall=base["yolo26s"]["recall"], f1=base["yolo26s"]["f1"]))
    with open(od / "benchmark_3way.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "radius_m", "precision", "recall", "f1"])
        w.writeheader(); w.writerows(out)
    print(f"wrote {od/'benchmark_3way.csv'}")

    m5 = mine[5.0]
    table = [("MY from-scratch heatmap CNN (s20)", m5["precision"], m5["recall"], m5["f1"]),
             ("DeepForest, fine-tuned (s19)", base["deepforest"]["precision"],
              base["deepforest"]["recall"], base["deepforest"]["f1"]),
             ("YOLO26s (Ahsan)", base["yolo26s"]["precision"], base["yolo26s"]["recall"],
              base["yolo26s"]["f1"])]

    print("\n========== HEAD-TO-HEAD (5 m distance match, labelled region) ==========")
    hdr = f"{'model':38s} {'P':>7} {'R':>7} {'F1':>7}"
    print(hdr); print("-" * len(hdr))
    for nm, p, r, f1 in sorted(table, key=lambda t: -t[3]):
        print(f"{nm:38s} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
    print("=" * len(hdr))

    labels = ["MY from-scratch\nheatmap CNN (s20)", "DeepForest\n(s19)", "YOLO26s\n(Ahsan)"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    xs = np.arange(3); wdt = 0.26
    for k, (metric, col) in enumerate(zip(["precision", "recall", "F1"],
                                          ["#a6cee3", "#fdb863", "#7b3294"])):
        vals = [t[1 + k] for t in table]
        b = ax.bar(xs + (k - 1) * wdt, vals, wdt, color=col, edgecolor="white", label=metric)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(1.0, max(t[3] for t in table) + 0.15))
    ax.set_ylabel("score (distance match, 5 m, labelled region)")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.set_title("Gables 3-way benchmark — same site, same protocol")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    radii = [r for r in cf["match_radii_m"]]
    ax.plot(radii, [mine[r]["f1"] for r in radii], "-o", color="#7b3294", ms=5,
            label="MY from-scratch (s20)")
    ax.plot([r for r in radii if r in df.index], [float(df.loc[r].f1) for r in radii if r in df.index],
            "-o", color="#1b7837", ms=5, label="DeepForest (s19)")
    ax.plot([5.0], [base["yolo26s"]["f1"]], "*", color="#4575b4", ms=16, label="YOLO26s (5 m only)")
    ax.set_xlabel("match radius (m)"); ax.set_ylabel("F1"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.set_title("F1 vs matching tolerance")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(rd / "benchmark_3way.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {rd/'benchmark_3way.png'}")
    return out


def verdict(cfg, rows, out):
    cf = gs(cfg); base = cf["baselines"]
    mine = {r["radius_m"]: r for r in rows if r["scope"] == "clean_eval_region"}
    m5 = mine[5.0]
    print("\n========================= VERDICT =========================")
    for r in cf["match_radii_m"]:
        m = mine[r]
        print(f"radius {r} m: P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}   "
              f"| random-scatter recall {m['rand_recall']:.3f}  edge {m['edge_recall']:+.3f}")
    rank = sorted([("from-scratch (mine)", m5["f1"]), ("DeepForest", base["deepforest"]["f1"]),
                   ("YOLO26s", base["yolo26s"]["f1"])], key=lambda t: -t[1])
    print(f"\n5 m head-to-head ranking: " +
          "  >  ".join(f"{n} {v:.3f}" for n, v in rank))
    d = m5["f1"] - base["deepforest"]["f1"]; y = m5["f1"] - base["yolo26s"]["f1"]
    print(f"from-scratch vs DeepForest: {d:+.3f} F1   |   vs YOLO26s: {y:+.3f} F1")
    print(f"peak RAM this process: {peak_ram_gb():.1f} GB")
    print("NOTE: distance eval (not IoU); the inventory is known to undercount, so")
    print("      precision is conservative for every model in this table equally.")
    print("===========================================================")


def main():
    cfg = C.load_config(); np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    od, rd = paths(cfg)

    if env_on("TD_SKIP_CHIPS") and (od / "chips" / "manifest.csv").exists():
        print("[stage1] TD_SKIP_CHIPS set — reusing the cached chips.")
    else:
        build_chips(cfg, od, rd)

    if env_on("TD_SKIP_TRAIN"):
        print("\n[stage2] TD_SKIP_TRAIN set — evaluating the existing checkpoint.")
    else:
        train(cfg, od, rd)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = od / "model" / "scratch_best.pt"
    if not ckpt.exists():
        ckpt = od / "model" / "scratch_last.pt"
    model = build_model(gs(cfg)["width"], gs(cfg)["out_stride"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"\nloaded {ckpt.name} for inference")

    raw_xy, raw_sc = predict_test(cfg, od, model, device)
    rows, hconf = evaluate(cfg, od, rd, raw_xy, raw_sc)
    fig_heatmap_chip(cfg, od, rd, model, device, hconf)
    fig_overlay(cfg, od, rd, raw_xy, raw_sc, hconf)
    out = benchmark(cfg, od, rd, rows)
    verdict(cfg, rows, out)


if __name__ == "__main__":
    main()
