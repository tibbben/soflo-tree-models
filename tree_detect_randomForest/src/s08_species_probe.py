"""
Step 08 — PALM vs NON-PALM feasibility probe (classification CEILING).  [local]

Decoupled from detection: we crop GROUND-TRUTH canopy boxes (s07 species labels)
straight from the orthomosaic and ask how well a binary classifier separates palm
(Arecaceae) from everything else. This is an UPPER BOUND — real end-to-end species
accuracy is lower, bounded by detection recall (a crown the detector misses can
never be classified). Binary only.

Pipeline (self-contained; nothing trained elsewhere depends on this):
  1. dataset   palm=1 / non-palm=0 from outputs/species/species_boxes.geojson;
               drop 'unknown' (missing family); pine/cypress fold into non-palm.
  2. crops     read each box window from the ortho, resize 160x160 RGB; skip
               nodata/black/unreadable windows; cached to outputs/species_probe/.
  3. split     SPATIAL x-band split (train <60th pct easting, val 60-80, test >=80)
               so adjacent crowns don't leak across splits.
  4. models    (a) linear probe: frozen ImageNet ResNet18 features + logistic reg.
               (b) fine-tune ResNet18 end-to-end (low LR, early stop, flip/rot/jitter aug).
  5. eval      test accuracy vs base rate, palm P/R/F1, confusion matrix, for BOTH.
  6. figures   confusion matrices, fine-tune curve, sample-crop grid (hits + misses).

Device mps / 32-bit / batch 32.  Model weights + crop cache are gitignored.
Run:  SOFLO_DATA_ROOT=/path/to/soflo_data .venv/bin/python src/s08_species_probe.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import csv
import time
from pathlib import Path
import numpy as np
import geopandas as gpd
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

CROP = 160
BATCH = 32
FT_EPOCHS = 20
FT_PATIENCE = 5
FT_LR = 1e-4
SEED = 0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── 1+3) dataset + spatial split ────────────────────────────────────────────
def build_split(cfg):
    b = gpd.read_file(C.p(cfg, cfg["outputs_dir"]) / "species" / "species_boxes.geojson")
    d = b[b["species_class"] != "unknown"].copy().reset_index(drop=True)
    d["palm"] = (d["species_class"] == "palm").astype(int)
    n = len(d); npalm = int(d["palm"].sum())
    base = max(npalm, n - npalm) / n
    print(f"binary dataset: {n} boxes  (dropped {len(b)-n} 'unknown')")
    print(f"  palm     = {npalm} ({npalm/n*100:.1f}%)")
    print(f"  non-palm = {n-npalm} ({(n-npalm)/n*100:.1f}%)  "
          f"[pine+cypress folded in]")
    print(f"  majority base rate = {base*100:.1f}%  (predict-all-non-palm accuracy)")

    x = d["cx"].values
    q60, q80 = np.quantile(x, [0.60, 0.80])
    split = np.where(x < q60, "train", np.where(x < q80, "val", "test"))
    d["split"] = split
    print("\nspatial x-band split (easting quantiles 0.60, 0.80):")
    for s in ("train", "val", "test"):
        m = d["split"] == s
        p = int(d.loc[m, "palm"].sum()); tot = int(m.sum())
        print(f"  {s:6s} n={tot:4d}  palm={p:4d} ({p/tot*100:4.1f}%)  non-palm={tot-p:4d}")
    return d


# ── 2) crop canopies from the ortho (with on-disk cache) ────────────────────
def load_crops(cfg, d):
    cache_dir = C.p(cfg, cfg["outputs_dir"]) / "species_probe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "crop_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["idx"]) and int(z["crop"][0].shape[0]) == CROP:
            print(f"\nloaded cached crops: {len(z['idx'])} from {cache.name}")
            return z["crop"], z["idx"]

    import rasterio
    from rasterio.windows import from_bounds
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    ortho = C.input_path(cfg, "orthomosaic")
    print(f"\ncropping {len(d)} boxes from {ortho.name} -> {CROP}x{CROP} ...")

    crops, kept_idx = [], []
    skip_read = skip_black = 0
    t0 = time.time()
    with rasterio.open(ortho) as src:
        nb = src.count
        for i, r in d.iterrows():
            try:
                win = from_bounds(r["minx"], r["miny"], r["maxx"], r["maxy"], src.transform)
                arr = src.read(indexes=[1, 2, 3] if nb >= 3 else [1, 1, 1],
                               window=win, boundless=True, fill_value=0)
            except Exception:
                skip_read += 1
                continue
            if arr.size == 0 or arr.shape[1] < 4 or arr.shape[2] < 4:
                skip_read += 1
                continue
            img = np.transpose(arr, (1, 2, 0))            # HWC uint8
            # nodata / black-edge guard: too many all-zero pixels => off-ortho
            black = float(np.mean(np.all(img == 0, axis=2)))
            if black > 0.30:
                skip_black += 1
                continue
            pil = Image.fromarray(img.astype(np.uint8)).resize((CROP, CROP), Image.BILINEAR)
            crops.append(np.asarray(pil, dtype=np.uint8))
            kept_idx.append(i)
            if (len(kept_idx) % 1000) == 0:
                print(f"  ...{len(kept_idx)} kept ({time.time()-t0:.0f}s)", flush=True)

    crop = np.stack(crops); idx = np.asarray(kept_idx)
    np.savez_compressed(cache, crop=crop, idx=idx)
    n_skip = skip_read + skip_black
    print(f"kept {len(idx)} crops; skipped {n_skip} "
          f"(unreadable/tiny={skip_read}, nodata/black={skip_black})")
    print(f"wrote cache {cache} ({cache.stat().st_size/1e6:.0f} MB, gitignored)")
    return crop, idx


# ── feature/model helpers ───────────────────────────────────────────────────
def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _norm_batch(crops_u8):
    """uint8 HWC stack -> float32 NCHW, ImageNet-normalized."""
    import torch
    x = crops_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())


def resnet18_backbone(pretrained=True):
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=w)
    feat = nn.Sequential(*list(m.children())[:-1])    # -> (N,512,1,1)
    return m, feat


# ── 4a) linear probe: frozen features + logistic regression ─────────────────
def linear_probe(crops, y, splits, device):
    import torch
    import torch.nn as nn
    _, feat = resnet18_backbone(pretrained=True)
    feat = feat.to(device).eval()

    def extract(mask):
        xs = _norm_batch(crops[mask])
        out = []
        with torch.no_grad():
            for j in range(0, len(xs), BATCH):
                fb = feat(xs[j:j+BATCH].to(device)).squeeze(-1).squeeze(-1)
                out.append(fb.cpu())
        return torch.cat(out) if out else torch.empty(0, 512)

    print("\n[linear probe] extracting frozen ResNet18 features ...", flush=True)
    feats = {s: extract(splits == s) for s in ("train", "val", "test")}
    yt = {s: torch.from_numpy(y[splits == s].astype(np.float32)) for s in ("train", "val", "test")}

    # standardize features (helps logistic reg), stats from train
    mu, sd = feats["train"].mean(0, keepdim=True), feats["train"].std(0, keepdim=True) + 1e-6
    Z = {s: (feats[s] - mu) / sd for s in feats}

    clf = nn.Linear(512, 1)
    pos_w = torch.tensor([(yt["train"] == 0).sum() / max(1, (yt["train"] == 1).sum())])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    best_val, best_state = -1, None
    for ep in range(300):
        clf.train(); opt.zero_grad()
        loss = lossf(clf(Z["train"]).squeeze(1), yt["train"])
        loss.backward(); opt.step()
        if ep % 10 == 0 or ep == 299:
            clf.eval()
            with torch.no_grad():
                vp = (clf(Z["val"]).squeeze(1) > 0).float()
                acc = (vp == yt["val"]).float().mean().item()
            if acc > best_val:
                best_val, best_state = acc, {k: v.clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        test_logits = clf(Z["test"]).squeeze(1)
    pred = (test_logits > 0).int().numpy()
    print(f"[linear probe] best val acc={best_val:.3f}")
    return pred


# ── 4b) fine-tune ResNet18 end-to-end ───────────────────────────────────────
def finetune(cfg, crops, y, splits, device):
    import torch
    import torch.nn as nn

    g = torch.Generator().manual_seed(SEED)
    tr = np.where(splits == "train")[0]
    va = np.where(splits == "val")[0]

    Xtr_u8 = crops[tr]; ytr = torch.from_numpy(y[tr].astype(np.float32))
    Xva = _norm_batch(crops[va]); yva = torch.from_numpy(y[va].astype(np.float32))

    def augment(batch_u8):
        """random h/v flip + 90° rotation (rotation-invariant from above) + mild jitter."""
        x = batch_u8.astype(np.float32) / 255.0
        out = np.empty_like(x)
        for i in range(len(x)):
            im = x[i]
            k = int(torch.randint(0, 4, (1,), generator=g).item())
            im = np.rot90(im, k=k, axes=(0, 1))
            if torch.rand(1, generator=g).item() < 0.5:
                im = im[::-1, :, :]
            if torch.rand(1, generator=g).item() < 0.5:
                im = im[:, ::-1, :]
            # mild brightness/contrast jitter
            br = 1.0 + (torch.rand(1, generator=g).item() - 0.5) * 0.3
            ct = 1.0 + (torch.rand(1, generator=g).item() - 0.5) * 0.3
            im = np.clip((im * br - 0.5) * ct + 0.5, 0, 1)
            out[i] = im
        out = (out - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.transpose(out, (0, 3, 1, 2)).copy())

    model, _ = resnet18_backbone(pretrained=True)
    model.fc = nn.Linear(512, 1)
    model = model.to(device)

    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())]).to(device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=1e-4)

    rows, best_acc, best_state, bad = [], -1.0, None, 0
    print(f"\n[fine-tune] ResNet18 end-to-end  device={device} lr={FT_LR} batch={BATCH} "
          f"max_epochs={FT_EPOCHS} patience={FT_PATIENCE} pos_weight={pos_w.item():.2f}", flush=True)
    for ep in range(FT_EPOCHS):
        t0 = time.time()
        perm = torch.randperm(len(tr), generator=g).numpy()
        model.train(); tot = 0.0
        for j in range(0, len(perm), BATCH):
            bidx = perm[j:j+BATCH]
            xb = augment(Xtr_u8[bidx]).to(device)
            yb = ytr[bidx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(1), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(bidx)
        train_loss = tot / len(perm)

        model.eval(); vtot = 0.0; correct = 0
        with torch.no_grad():
            for j in range(0, len(va), BATCH):
                xb = Xva[j:j+BATCH].to(device); yb = yva[j:j+BATCH].to(device)
                lo = model(xb).squeeze(1)
                vtot += lossf(lo, yb).item() * len(yb)
                correct += ((lo > 0).float() == yb).sum().item()
        val_loss = vtot / len(va); val_acc = correct / len(va)
        secs = time.time() - t0
        rows.append({"epoch": ep, "train_loss": round(train_loss, 4),
                     "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
                     "seconds": round(secs, 1)})
        print(f"  [epoch {ep:02d}] {secs:5.1f}s  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                print(f"  early stop (no val_acc gain in {FT_PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    wpath = C.p(cfg, cfg["outputs_dir"]) / "species_probe" / "finetune_best.pt"
    torch.save(best_state, wpath)
    print(f"[fine-tune] best val_acc={best_acc:.3f}  saved {wpath.name} (gitignored)")

    te = np.where(splits == "test")[0]
    Xte = _norm_batch(crops[te])
    model.eval(); preds = []
    with torch.no_grad():
        for j in range(0, len(Xte), BATCH):
            lo = model(Xte[j:j+BATCH].to(device)).squeeze(1)
            preds.append((lo > 0).int().cpu().numpy())
    return np.concatenate(preds), rows


# ── 5) metrics ──────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(accuracy=acc, palm_precision=prec, palm_recall=rec, palm_f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    cfg = C.load_config()
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    device = _device()

    d = build_split(cfg)
    crops, idx = load_crops(cfg, d)
    d = d.loc[idx].reset_index(drop=True)          # align rows to kept crops
    y = d["palm"].values.astype(int)
    splits = d["split"].values
    base_rate = max((y == 1).mean(), (y == 0).mean())

    lp_pred = linear_probe(crops, y, splits, device)
    ft_pred, ft_rows = finetune(cfg, crops, y, splits, device)

    y_test = y[splits == "test"]
    results = {"linear_probe": metrics(y_test, lp_pred),
               "finetune": metrics(y_test, ft_pred)}
    print(f"\n===== TEST results (base rate = {base_rate*100:.1f}%) =====")
    for name, m in results.items():
        print(f"  {name:13s} acc={m['accuracy']*100:5.1f}%  "
              f"palm P/R/F1={m['palm_precision']:.3f}/{m['palm_recall']:.3f}/{m['palm_f1']:.3f}  "
              f"(TP{m['tp']} FP{m['fp']} FN{m['fn']} TN{m['tn']})")

    _write_metrics(cfg, results, base_rate, ft_rows, splits, y)
    _fig_confusion(cfg, results, base_rate)
    _fig_curve(cfg, ft_rows)
    _fig_samples(cfg, crops, d, y_test, ft_pred, splits)
    _report(results, base_rate)


def _write_metrics(cfg, results, base_rate, ft_rows, splits, y):
    sd = C.p(cfg, cfg["outputs_dir"]) / "species_probe"
    out = sd / "probe_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_accuracy", "base_rate", "lift_over_base",
                    "palm_precision", "palm_recall", "palm_f1", "tp", "fp", "fn", "tn"])
        for name, m in results.items():
            w.writerow([name, round(m["accuracy"], 4), round(base_rate, 4),
                        round(m["accuracy"] - base_rate, 4),
                        round(m["palm_precision"], 4), round(m["palm_recall"], 4),
                        round(m["palm_f1"], 4), m["tp"], m["fp"], m["fn"], m["tn"]])
    print(f"wrote {out}")
    curve = sd / "finetune_curve.csv"
    with open(curve, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "seconds"])
        w.writeheader(); w.writerows(ft_rows)
    print(f"wrote {curve}")


def _fig_confusion(cfg, results, base_rate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() * 0.5 else "#222")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred non-palm", "pred palm"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true non-palm", "true palm"])
        ax.set_title(f"{name}\nacc={m['accuracy']*100:.1f}%  (base {base_rate*100:.1f}%)  "
                     f"palm F1={m['palm_f1']:.2f}")
    fig.suptitle("s08 — palm-vs-non-palm confusion (held-out spatial TEST split)")
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "species_probe" / "confusion_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _fig_curve(cfg, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    if not rows:
        return
    ep = [r["epoch"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, [r["train_loss"] for r in rows], "-o", color="#1b7837", ms=3, label="train loss")
    ax1.plot(ep, [r["val_loss"] for r in rows], "-o", color="#d73027", ms=3, label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(ep, [r["val_acc"] for r in rows], "-s", color="#4575b4", ms=3, label="val acc")
    ax2.set_ylabel("val accuracy"); ax2.legend(loc="upper right")
    be = max(rows, key=lambda r: r["val_acc"])
    ax2.axvline(be["epoch"], color="#999", ls="--", lw=1)
    ax1.set_title(f"s08 — fine-tune ResNet18 (best val_acc={be['val_acc']:.3f} @ epoch {be['epoch']})")
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "species_probe" / "finetune_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _fig_samples(cfg, crops, d, y_test, ft_pred, splits):
    """Grid: correct palms, correct non-palms, and misclassified (fine-tune)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    test_pos = np.where(splits == "test")[0]      # positions into crops/d for test rows
    correct_palm = test_pos[(ft_pred == 1) & (y_test == 1)]
    correct_non = test_pos[(ft_pred == 0) & (y_test == 0)]
    wrong = test_pos[ft_pred != y_test]

    cols = 6
    groups = [("correct palm", correct_palm, "#1b7837"),
              ("correct non-palm", correct_non, "#333333"),
              ("MISCLASSIFIED", wrong, "#d73027")]
    fig, axes = plt.subplots(3, cols, figsize=(2.0 * cols, 6.6))
    rng = np.random.default_rng(SEED)
    for row, (label, pool, color) in enumerate(groups):
        pick = rng.choice(pool, size=min(cols, len(pool)), replace=False) if len(pool) else []
        for c in range(cols):
            ax = axes[row, c]; ax.axis("off")
            if c < len(pick):
                pos = pick[c]
                ax.imshow(crops[pos])
                t = "palm" if y_test[np.where(test_pos == pos)[0][0]] == 1 else "non-palm"
                p = "palm" if ft_pred[np.where(test_pos == pos)[0][0]] == 1 else "non-palm"
                ttl = t if row < 2 else f"true {t}\npred {p}"
                ax.set_title(ttl, fontsize=8, color=color)
            if c == 0:
                ax.text(-0.12, 0.5, label, rotation=90, transform=ax.transAxes,
                        va="center", ha="center", fontsize=10, color=color, weight="bold")
    fig.suptitle("s08 — sample GT crops on TEST split (fine-tune predictions)", fontsize=12)
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "species_probe" / "sample_crops.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _report(results, base_rate):
    ft = results["finetune"]; lp = results["linear_probe"]
    best = max(results.values(), key=lambda m: m["accuracy"])
    lift = best["accuracy"] - base_rate
    print("\n========================= VERDICT =========================")
    print(f"base rate (predict-all-majority) : {base_rate*100:.1f}%")
    print(f"linear probe test accuracy       : {lp['accuracy']*100:.1f}%")
    print(f"fine-tune   test accuracy        : {ft['accuracy']*100:.1f}%")
    print(f"best lift over base rate         : +{lift*100:.1f} pts")
    verdict = "FEASIBLE" if lift > 0.10 else ("MARGINAL" if lift > 0.03 else "NOT useful")
    print(f"palm precision/recall (fine-tune): {ft['palm_precision']:.2f}/{ft['palm_recall']:.2f}")
    err = "false negatives (palms called non-palm)" if ft["fn"] >= ft["fp"] else \
          "false positives (non-palms called palm)"
    print(f"errors concentrate in            : {err}  (FN={ft['fn']} FP={ft['fp']})")
    print(f"==> {verdict}: palm-vs-non-palm beats base rate by +{lift*100:.0f} pts.")
    print("NOTE: this is the GROUND-TRUTH-CROP CEILING. Real end-to-end species")
    print("accuracy will be LOWER — it is bounded by detection recall (any crown")
    print("the detector misses is never classified at all).")
    print("===========================================================")


if __name__ == "__main__":
    main()
