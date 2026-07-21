"""
Step 13 — BCNP 3-class species probe: pine / cypress / palm.  [local, time-boxed]

Classification ceiling on GROUND-TRUTH census points from the 8 clean BCNP plots.
Decoupled from detection; asks how separable the three dominant species are from
canopy crops. Time-boxed and STAGED so partial results survive:

  stage 1  build crops (georeference alive pine/cypress/palm, crop RGB, resize) and
           WRITE per-class counts + base rate + bar figure  <- always produced
  stage 2  spatial split by PLOT (2 whole plots held out for test)
  stage 3  FAST linear probe (frozen ResNet18 features + logistic regression) + eval
           WRITE result + confusion  <- produced before any fine-tune
  stage 4  (bonus) fine-tune ResNet18 3-class, inverse-freq weighting, aug, early stop
  stage 5  figures + verdict

Species map: Pinus elliottii=pine, Taxodium ascendens=cypress, Sabal palmetto=palm.
All other species and dead trees are dropped. Crops from RGB (bands 1-3); band 4 is alpha.
Writes only under outputs/bcnp_species + reports/bcnp_species. Crop cache is gitignored.

Run:  .venv/bin/python src/s13_bcnp_species_probe.py
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
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

BC = Path("~/Downloads/bcnp").expanduser()
CLEAN = ["plot_17_3", "plot_1_1", "plot_3_1", "plot_9_3",
         "plot_12_1", "plot_10_3", "plot_6_2", "plot_18_2"]
TEST_PLOTS = ["plot_9_3", "plot_12_1"]          # held out; both have all 3 classes
SP2CLS = {"Pinus elliottii": "pine", "Taxodium ascendens": "cypress",
          "Sabal palmetto": "palm"}
CLASSES = ["pine", "cypress", "palm"]
CLS_COLOR = {"pine": "#31a354", "cypress": "#4575b4", "palm": "#f1a340"}
CROP = 128
BOX_M = 1.5
BATCH = 64
FT_EPOCHS = 15
FT_PATIENCE = 4
FT_LR = 1e-4
SEED = 0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _od(cfg):
    od = C.p(cfg, cfg["outputs_dir"]) / "bcnp_species"
    rd = C.p(cfg, cfg["reports_dir"]) / "bcnp_species"
    od.mkdir(parents=True, exist_ok=True); rd.mkdir(parents=True, exist_ok=True)
    return od, rd


# ── stage 1: build crops ────────────────────────────────────────────────────
def build_crops(cfg, od):
    cache = od / "crop_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"[stage1] loaded cached crops: {len(z['y'])}")
        return z["X"], z["y"], z["plot"]

    import rasterio
    from rasterio.windows import from_bounds
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    census = pd.read_csv(BC / "RP.Plot_census_data.2025.csv")
    polys = gpd.read_file(BC / "tree_plots_polygons.geojson").to_crs(32617)

    X, y, plotcol = [], [], []
    skips = {}
    print("[stage1] building crops from 8 clean plots ...", flush=True)
    for name in CLEAN:
        u, up = map(int, name.split("_")[1:])
        sub = census[(census.UNIT == u) & (census.UNITPLOT == up) & (census.ALIVE == 1)].copy()
        sub = sub[sub.SP.isin(SP2CLS)].dropna(subset=["XCOORD", "YCOORD"])
        poly = polys[polys.Name == name].geometry.iloc[0]
        minx, miny = poly.bounds[0], poly.bounds[1]
        tif = BC / f"{name}.tif"
        sk = 0
        with rasterio.open(tif) as src:
            gsd = abs(src.transform.a)
            half = (BOX_M / gsd) / 2.0 * gsd     # half-box in metres
            for _, r in sub.iterrows():
                wx, wy = minx + r.XCOORD, miny + r.YCOORD
                win = from_bounds(wx - half, wy - half, wx + half, wy + half, src.transform)
                try:
                    arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                except Exception:
                    sk += 1; continue
                if arr.shape[1] < 4 or arr.shape[2] < 4:
                    sk += 1; continue
                img = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
                if np.mean(np.all(img == 0, axis=2)) > 0.30:     # nodata/black
                    sk += 1; continue
                pil = Image.fromarray(img).resize((CROP, CROP), Image.BILINEAR)
                X.append(np.asarray(pil, dtype=np.uint8))
                y.append(SP2CLS[r.SP]); plotcol.append(name)
        skips[name] = sk
        print(f"   {name}: kept {sum(1 for p in plotcol if p==name)}  skipped {sk}", flush=True)
    X = np.stack(X); y = np.array(y); plotcol = np.array(plotcol)
    np.savez_compressed(cache, X=X, y=y, plot=plotcol)
    print(f"[stage1] total crops {len(y)}  skipped {sum(skips.values())}  "
          f"cache {cache.stat().st_size/1e6:.0f} MB (gitignored)")
    return X, y, plotcol


def stage1_report(cfg, od, rd, y, plotcol):
    n = len(y)
    counts = {c: int((y == c).sum()) for c in CLASSES}
    base = max(counts.values()) / n
    print("\n===== STAGE 1 — crop counts (alive pine/cypress/palm) =====")
    for c in CLASSES:
        print(f"  {c:8s} {counts[c]:6d}  ({counts[c]/n*100:4.1f}%)")
    print(f"  {'TOTAL':8s} {n:6d}")
    print(f"  base rate (majority = pine): {base*100:.1f}%")
    # per plot x class
    rows = []
    for pl in CLEAN:
        m = plotcol == pl
        rows.append(dict(plot=pl, total=int(m.sum()),
                         **{c: int(((y == c) & m).sum()) for c in CLASSES}))
    rows.append(dict(plot="TOTAL", total=n, **counts))
    with open(od / "crop_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["plot", "total"] + CLASSES); w.writeheader(); w.writerows(rows)
    print(f"wrote {od/'crop_counts.csv'}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [counts[c] for c in CLASSES]
    bars = ax.bar(CLASSES, vals, color=[CLS_COLOR[c] for c in CLASSES], width=.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+n*0.01, f"{v}\n{v/n*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("alive tree crops"); ax.set_ylim(0, max(vals)*1.15)
    ax.set_title(f"s13 — BCNP crop counts by class (n={n}, base rate {base*100:.0f}%)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(rd/"class_counts.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'class_counts.png'}")
    return counts, base


# ── stage 2: spatial split ──────────────────────────────────────────────────
def split(y, plotcol):
    test = np.isin(plotcol, TEST_PLOTS)
    train = ~test
    print("\n===== STAGE 2 — spatial split by PLOT =====")
    print(f"  test plots : {TEST_PLOTS}")
    for nm, m in [("train", train), ("test", test)]:
        cc = {c: int(((y == c) & m).sum()) for c in CLASSES}
        print(f"  {nm:5s} n={int(m.sum()):6d}  " + "  ".join(f"{c}={cc[c]}" for c in CLASSES))
    return train, test


# ── model helpers ───────────────────────────────────────────────────────────
def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _norm(Xu8):
    import torch
    x = Xu8.astype(np.float32)/255.0
    x = (x - IMAGENET_MEAN)/IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())


def _backbone(pretrained=True):
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    feat = nn.Sequential(*list(m.children())[:-1])
    return m, feat


def _metrics(y_true, y_pred):
    """3-class accuracy, per-class P/R, 3x3 confusion (rows=true, cols=pred)."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    yt = np.array([idx[c] for c in y_true]); yp = np.array([idx[c] for c in y_pred])
    cm = np.zeros((3, 3), int)
    for t, p in zip(yt, yp): cm[t, p] += 1
    acc = np.trace(cm)/cm.sum()
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; prec = tp/cm[:, i].sum() if cm[:, i].sum() else 0
        rec = tp/cm[i, :].sum() if cm[i, :].sum() else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        per[c] = dict(precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3))
    return dict(accuracy=round(float(acc), 4), per_class=per, cm=cm.tolist())


# ── stage 3: linear probe ───────────────────────────────────────────────────
def linear_probe(X, y, train, test, device):
    import torch, torch.nn as nn
    _, feat = _backbone(True); feat = feat.to(device).eval()
    idx = {c: i for i, c in enumerate(CLASSES)}

    def extract(mask):
        xs = _norm(X[mask]); out = []
        with torch.no_grad():
            for j in range(0, len(xs), BATCH):
                out.append(feat(xs[j:j+BATCH].to(device)).squeeze(-1).squeeze(-1).cpu())
        return torch.cat(out)
    print("\n[stage3] extracting frozen ResNet18 features ...", flush=True)
    Ftr, Fte = extract(train), extract(test)
    ytr = torch.tensor([idx[c] for c in y[train]])
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True)+1e-6
    Ztr, Zte = (Ftr-mu)/sd, (Fte-mu)/sd
    w = torch.tensor([len(ytr)/(3*(ytr == i).sum().clamp(min=1)) for i in range(3)]).float()
    clf = nn.Linear(512, 3); opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(400):
        clf.train(); opt.zero_grad(); lossf(clf(Ztr), ytr).backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Zte).argmax(1).numpy()
    return [CLASSES[p] for p in pred]


# ── stage 4: fine-tune ──────────────────────────────────────────────────────
def finetune(cfg, od, X, y, train, test, device):
    import torch, torch.nn as nn
    idx = {c: i for i, c in enumerate(CLASSES)}
    g = torch.Generator().manual_seed(SEED)
    tr = np.where(train)[0]; te = np.where(test)[0]
    Xtr = X[tr]; ytr = torch.tensor([idx[c] for c in y[tr]])
    Xte = _norm(X[te]); yte = torch.tensor([idx[c] for c in y[te]])

    def augment(bu8):
        x = bu8.astype(np.float32)/255.0; out = np.empty_like(x)
        for i in range(len(x)):
            im = x[i]; k = int(torch.randint(0, 4, (1,), generator=g).item())
            im = np.rot90(im, k, (0, 1))
            if torch.rand(1, generator=g).item() < .5: im = im[::-1]
            if torch.rand(1, generator=g).item() < .5: im = im[:, ::-1]
            br = 1+(torch.rand(1, generator=g).item()-.5)*.3
            im = np.clip(im*br, 0, 1); out[i] = im
        out = (out-IMAGENET_MEAN)/IMAGENET_STD
        return torch.from_numpy(np.transpose(out, (0, 3, 1, 2)).copy())

    model, _ = _backbone(True); model.fc = nn.Linear(512, 3); model = model.to(device)
    w = torch.tensor([len(ytr)/(3*(ytr == i).sum().clamp(min=1)) for i in range(3)]).float().to(device)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    rows, best, best_state, bad = [], -1, None, 0
    print(f"\n[stage4] fine-tune ResNet18 3-class  device={device} lr={FT_LR} batch={BATCH} "
          f"max_epochs={FT_EPOCHS} weights={[round(x,2) for x in w.tolist()]}", flush=True)
    for ep in range(FT_EPOCHS):
        t0 = time.time(); perm = torch.randperm(len(tr), generator=g).numpy()
        model.train(); tot = 0
        for j in range(0, len(perm), BATCH):
            bi = perm[j:j+BATCH]; xb = augment(Xtr[bi]).to(device); yb = ytr[bi].to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*len(bi)
        model.eval(); correct = 0; vloss = 0
        with torch.no_grad():
            for j in range(0, len(te), BATCH):
                xb = Xte[j:j+BATCH].to(device); yb = yte[j:j+BATCH].to(device)
                lo = model(xb); vloss += lossf(lo, yb).item()*len(yb)
                correct += (lo.argmax(1) == yb).sum().item()
        vacc = correct/len(te); secs = time.time()-t0
        rows.append(dict(epoch=ep, train_loss=round(tot/len(perm), 4),
                         val_loss=round(vloss/len(te), 4), val_acc=round(vacc, 4), seconds=round(secs, 1)))
        print(f"   [epoch {ep:02d}] {secs:5.1f}s train_loss={tot/len(perm):.4f} val_acc={vacc:.4f}", flush=True)
        _write_curve(od, rows)                    # incremental so partial survives
        if vacc > best: best, bad, best_state = vacc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= FT_PATIENCE: print("   early stop"); break
    model.load_state_dict(best_state)
    torch.save(best_state, od/"finetune_best.pt")
    model.eval(); preds = []
    with torch.no_grad():
        for j in range(0, len(Xte), BATCH):
            preds.append(model(Xte[j:j+BATCH].to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return [CLASSES[p] for p in pred], rows


def _write_curve(od, rows):
    with open(od/"finetune_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "seconds"])
        w.writeheader(); w.writerows(rows)


# ── result writers / figures ────────────────────────────────────────────────
def write_result(od, model, m, base):
    p = od/"probe_results.csv"; new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["model", "test_accuracy", "base_rate"] +
                           [f"{c}_{k}" for c in CLASSES for k in ("precision", "recall", "f1")])
        row = [model, m["accuracy"], round(base, 4)]
        for c in CLASSES:
            row += [m["per_class"][c]["precision"], m["per_class"][c]["recall"], m["per_class"][c]["f1"]]
        w.writerow(row)
    print(f"wrote {p}  [{model}]")


def fig_confusion(rd, model, m, base):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    cm = np.array(m["cm"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center",
                color="white" if v > cm.max()*.5 else "#222", fontsize=12)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"pred {c}" for c in CLASSES], fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"true {c}" for c in CLASSES], fontsize=9)
    ax.set_title(f"s13 — {model}\nacc {m['accuracy']*100:.1f}% (base {base*100:.0f}%)")
    fig.tight_layout(); fig.savefig(rd/f"confusion_{model}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/f'confusion_{model}.png'}")


def fig_curve(rd, rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    if not rows: return
    ep = [r["epoch"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, [r["train_loss"] for r in rows], "-o", color="#1b7837", ms=3, label="train loss")
    ax1.plot(ep, [r["val_loss"] for r in rows], "-o", color="#d73027", ms=3, label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("CE loss"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx(); ax2.plot(ep, [r["val_acc"] for r in rows], "-s", color="#4575b4", ms=3, label="val acc")
    ax2.set_ylabel("val accuracy"); ax2.legend(loc="upper right")
    ax1.set_title("s13 — fine-tune ResNet18 (BCNP 3-class)")
    fig.tight_layout(); fig.savefig(rd/"finetune_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'finetune_curve.png'}")


def fig_samples(rd, X, y):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    rng = np.random.default_rng(SEED)
    cols = 6
    fig, axes = plt.subplots(3, cols, figsize=(2*cols, 6.4))
    for row, c in enumerate(CLASSES):
        pool = np.where(y == c)[0]
        pick = rng.choice(pool, min(cols, len(pool)), replace=False)
        for k in range(cols):
            ax = axes[row, k]; ax.axis("off")
            if k < len(pick): ax.imshow(X[pick[k]])
            if k == 0:
                ax.text(-0.12, 0.5, c, rotation=90, transform=ax.transAxes, va="center",
                        ha="center", fontsize=11, color=CLS_COLOR[c], weight="bold")
    fig.suptitle("s13 — BCNP sample crops per class (GT census)", fontsize=12)
    fig.tight_layout(); fig.savefig(rd/"sample_crops.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'sample_crops.png'}")


def verdict(base, lin_m, ft_m):
    print("\n========================= VERDICT =========================")
    print(f"base rate (predict-all-pine): {base*100:.1f}%")
    print(f"linear probe test accuracy  : {lin_m['accuracy']*100:.1f}%")
    if ft_m: print(f"fine-tune   test accuracy   : {ft_m['accuracy']*100:.1f}%")
    best = ft_m or lin_m
    lift = best["accuracy"]-base
    cyr = best["per_class"]["cypress"]["recall"]
    cm = np.array(best["cm"]); ci = CLASSES.index("cypress"); pi = CLASSES.index("pine")
    cy_as_pine = cm[ci, pi]/cm[ci].sum() if cm[ci].sum() else 0
    print(f"lift over base rate         : {lift*100:+.1f} pts")
    print(f"cypress recall              : {cyr:.2f}  ({cm[ci,pi]}/{cm[ci].sum()} cypress -> predicted pine = {cy_as_pine*100:.0f}%)")
    print(f"palm recall                 : {best['per_class']['palm']['recall']:.2f}")
    v = "BEATS base rate" if lift > 0.03 else "does NOT clearly beat base rate"
    print(f"==> {v}. Cypress is {'largely lost into pine' if cy_as_pine>0.4 else 'partially recovered'}.")
    print("NOTE: GT-crop ceiling; dense stands mean crops overlap neighbours. Real end-to-end")
    print("species accuracy will be lower, bounded by detection recall.")
    print("===========================================================")


def main():
    cfg = C.load_config(); np.random.seed(SEED)
    import torch; torch.manual_seed(SEED)
    device = _device()
    od, rd = _od(cfg)

    X, y, plotcol = build_crops(cfg, od)
    counts, base = stage1_report(cfg, od, rd, y, plotcol)      # stage 1 artifacts
    fig_samples(rd, X, y)
    train, test = split(y, plotcol)                            # stage 2

    lin_pred = linear_probe(X, y, train, test, device)         # stage 3
    lin_m = _metrics(y[test], lin_pred)
    print("\n===== STAGE 3 — linear probe (frozen ResNet18 + logreg) =====")
    print(f"  test acc {lin_m['accuracy']*100:.1f}%  (base {base*100:.1f}%)")
    for c in CLASSES: print(f"    {c:8s} P/R/F1 = {lin_m['per_class'][c]['precision']:.2f}/"
                            f"{lin_m['per_class'][c]['recall']:.2f}/{lin_m['per_class'][c]['f1']:.2f}")
    write_result(od, "linear_probe", lin_m, base); fig_confusion(rd, "linear_probe", lin_m, base)

    ft_m = None
    try:                                                       # stage 4 (bonus)
        ft_pred, ft_rows = finetune(cfg, od, X, y, train, test, device)
        ft_m = _metrics(y[test], ft_pred)
        print("\n===== STAGE 4 — fine-tune ResNet18 =====")
        print(f"  test acc {ft_m['accuracy']*100:.1f}%  (base {base*100:.1f}%)")
        for c in CLASSES: print(f"    {c:8s} P/R/F1 = {ft_m['per_class'][c]['precision']:.2f}/"
                                f"{ft_m['per_class'][c]['recall']:.2f}/{ft_m['per_class'][c]['f1']:.2f}")
        write_result(od, "finetune", ft_m, base)
        fig_confusion(rd, "finetune", ft_m, base); fig_curve(rd, ft_rows)
    except Exception as e:
        print(f"\n[stage4] fine-tune skipped/failed ({e}) — linear-probe result stands.")

    verdict(base, lin_m, ft_m)


if __name__ == "__main__":
    main()
