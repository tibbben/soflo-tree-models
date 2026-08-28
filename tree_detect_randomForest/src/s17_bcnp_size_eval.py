"""
Step 17 — Is the BCNP recall ceiling label VISIBILITY?  [local, NO retraining]

Re-scores the EXISTING s16 YOLO predictions (same weights, same operating point)
against size-filtered census subsets. If recall (and edge over random) climbs
sharply as we keep only larger/taller trees, the misses are small/invisible stems
we cannot see in RGB — a label-visibility ceiling, not a model failure. If recall
stays flat, the model genuinely is not finding even the canopy-sized trees.

  1  census DBH / TOTHT distributions for the two test plots (hist, quartiles, stems/ha)
  2  re-run the distance eval (1.0 & 1.5 m + random-scatter floor) against subsets:
     all alive; DBH > p50/p75/p90; TOTHT > p50/p75/p90
     -> model P/R/F1, random floor, edge, target count, predicted count
  3/4 recall + edge trend vs size; predicted-vs-target count per subset
  5  figures: DBH/TOTHT histograms, recall-vs-percentile curves, predictions-vs-
     large-tree-only census overlay

NO training and NO new labels: loads s16's final weights and re-uses its predictions
(cached to disk on first run). DBH is 100% populated; TOTHT is only ~17% populated
(height was recorded for a non-random minority) — TOTHT subsets are flagged as such.

Run:  .venv/bin/python src/s17_bcnp_size_eval.py
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

SEED = 0
PCTS = [50, 75, 90]


def _od(cfg):
    od = C.p(cfg, cfg["outputs_dir"]) / "bcnp_yolo"          # reuse s16's dirs
    rd = C.p(cfg, cfg["reports_dir"]) / "bcnp_yolo"
    od.mkdir(parents=True, exist_ok=True); rd.mkdir(parents=True, exist_ok=True)
    return od, rd


def census_full(cfg, name, census, polys):
    """Per-plot ALIVE trees with world (x,y), DBH, TOTHT, and plot area (ha)."""
    u, up = Y._plot_uid(name)
    s = census[(census.UNIT == u) & (census.UNITPLOT == up) & (census.ALIVE == 1)].dropna(subset=["XCOORD", "YCOORD"])
    poly = polys[polys.Name == name].to_crs(cfg["crs"]).geometry.iloc[0]
    swx, swy = poly.bounds[0], poly.bounds[1]
    df = pd.DataFrame({
        "x": swx + pd.to_numeric(s.XCOORD, errors="coerce").values,
        "y": swy + pd.to_numeric(s.YCOORD, errors="coerce").values,
        "dbh": pd.to_numeric(s.DBH, errors="coerce").values,
        "totht": pd.to_numeric(s.TOTHT, errors="coerce").values,
    })
    return df, poly.area / 1e4


# ── stage 1: distributions ───────────────────────────────────────────────────
def distributions(cfg, rd, cens):
    print("\n===== STAGE 1 — census size distributions (ALIVE, test plots) =====")
    rows = []
    for name, (df, ha) in cens.items():
        dbh = df.dbh.dropna(); tot = df.totht.dropna()
        q = lambda a, p: round(float(np.percentile(a, p)), 1) if len(a) else None
        print(f"  {name}: alive={len(df)}  area={ha:.2f} ha  stems/ha={len(df)/ha:.0f}")
        print(f"    DBH   n={len(dbh):5d} (NA {df.dbh.isna().sum()})  "
              f"q25/50/75/90 = {q(dbh,25)}/{q(dbh,50)}/{q(dbh,75)}/{q(dbh,90)} cm")
        print(f"    TOTHT n={len(tot):5d} (NA {df.totht.isna().sum()}, {len(tot)/len(df)*100:.0f}% coverage)  "
              f"q25/50/75/90 = {q(tot,25)}/{q(tot,50)}/{q(tot,75)}/{q(tot,90)} m")
        rows.append(dict(plot=name, alive=len(df), stems_per_ha=round(len(df)/ha),
                         dbh_n=len(dbh), dbh_p50=q(dbh, 50), dbh_p75=q(dbh, 75), dbh_p90=q(dbh, 90),
                         totht_n=len(tot), totht_cov=round(len(tot)/len(df), 3),
                         totht_p50=q(tot, 50), totht_p75=q(tot, 75), totht_p90=q(tot, 90)))
    od, _ = _od(cfg)
    with open(od / "size_distributions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    _fig_hist(rd, cens)
    return rows


def _fig_hist(rd, cens):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    alldbh = np.concatenate([df.dbh.dropna().values for df, _ in cens.values()])
    alltot = np.concatenate([df.totht.dropna().values for df, _ in cens.values()])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.hist(alldbh, bins=40, color="#4575b4", alpha=0.85)
    for p in PCTS:
        v = np.percentile(alldbh, p); a1.axvline(v, color="#d73027", ls="--", lw=1)
        a1.text(v, a1.get_ylim()[1]*0.92, f"p{p}={v:.0f}", fontsize=8, rotation=90, va="top")
    a1.set_xlabel("DBH (cm)"); a1.set_ylabel("alive trees"); a1.set_title(f"DBH (test plots, n={len(alldbh)}, 100% coverage)")
    a2.hist(alltot, bins=30, color="#1b7837", alpha=0.85)
    for p in PCTS:
        v = np.percentile(alltot, p); a2.axvline(v, color="#d73027", ls="--", lw=1)
        a2.text(v, a2.get_ylim()[1]*0.92, f"p{p}={v:.1f}", fontsize=8, rotation=90, va="top")
    a2.set_xlabel("TOTHT (m)"); a2.set_ylabel("alive trees")
    a2.set_title(f"TOTHT (test plots, n={len(alltot)}, ~{len(alltot)/sum(len(df) for df,_ in cens.values())*100:.0f}% coverage)")
    fig.suptitle("s17 — BCNP census size distributions (held-out test plots)", fontsize=12)
    fig.tight_layout(); fig.savefig(rd/"size_distributions.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'size_distributions.png'}")


# ── predictions (s16 model, cached; inference only) ──────────────────────────
def get_predictions(cfg, cens, census, polys):
    od, _ = _od(cfg)
    cache = od / "test_raw_predictions.npz"
    conf_floor = 0.02
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        raw = {k: dict(xy=z[f"{k}_xy"], sc=z[f"{k}_sc"]) for k in cfg["bcnp_yolo"]["test_plots"]}
        print(f"[pred] loaded cached s16 predictions ({sum(len(v['xy']) for v in raw.values())} raw dets)")
        return raw
    bc = json.loads((od / "best_config.json").read_text())
    best_pt = od / "runs" / "final" / "weights" / "best.pt"
    print(f"[pred] running s16 model inference (no training): {best_pt.name}, gsd {bc['gsd_m']*100:.0f}cm", flush=True)
    raw, save = {}, {}
    for name in cfg["bcnp_yolo"]["test_plots"]:
        xy, sc = Y.predict_plot_raw(cfg, name, best_pt, bc["gsd_m"], census, polys, conf_floor=conf_floor)
        raw[name] = dict(xy=xy, sc=sc); save[f"{name}_xy"] = xy; save[f"{name}_sc"] = sc
        print(f"   {name}: {len(xy)} raw detections (conf>={conf_floor})", flush=True)
    np.savez_compressed(cache, **save)
    return raw


def pred_set(cfg, raw, name, conf):
    """The model's prediction set at a confidence, NMS-merged (matches s16 protocol)."""
    m = raw[name]["sc"] >= conf
    return Y._nms_world(raw[name]["xy"][m], raw[name]["sc"][m], cfg["bcnp_yolo"]["peak_min_dist_m"])


# ── stage 2: re-score against size subsets ───────────────────────────────────
def subsets(cens):
    """Return list of (label, field, pct, per-plot filter fn). Percentiles are over
    the COMBINED test set (DBH over all; TOTHT over trees WITH height)."""
    alldbh = np.concatenate([df.dbh.dropna().values for df, _ in cens.values()])
    alltot = np.concatenate([df.totht.dropna().values for df, _ in cens.values()])
    out = [("all_alive", None, None, None)]
    for p in PCTS:
        thr = float(np.percentile(alldbh, p))
        out.append((f"DBH>p{p}", "dbh", p, ("dbh", thr)))
    for p in PCTS:
        thr = float(np.percentile(alltot, p))
        out.append((f"TOTHT>p{p}", "totht", p, ("totht", thr)))
    return out


def _subset_xy(df, spec):
    if spec is None:
        return df[["x", "y"]].values
    field, thr = spec
    s = df[df[field] > thr]
    return s[["x", "y"]].values


def evaluate(cfg, rd, cens, raw, polys):
    cf = cfg["bcnp_yolo"]; radii = cf["match_radii_m"]; conf = 0.02
    preds = {n: pred_set(cfg, raw, n, conf) for n in cf["test_plots"]}
    n_pred_total = sum(len(preds[n][0]) for n in cf["test_plots"])
    subs = subsets(cens)

    rows = []
    print(f"\n===== STAGE 2 — re-score s16 predictions (conf {conf}, {n_pred_total} preds) vs size subsets =====")
    hdr = (f"{'subset':11s} {'target':>6} {'pred':>6} | "
           f"{'r':>4} {'MOD_P':>6} {'MOD_R':>6} {'MOD_F1':>6} {'RND_R':>6} {'edgeR':>6}")
    print(hdr); print("-" * len(hdr))
    for label, field, pct, spec in subs:
        # combined per-plot GT for this subset
        per_plot = {}
        for n in cf["test_plots"]:
            per_plot[n] = dict(pred=preds[n][0], score=preds[n][1], gt=_subset_xy(cens[n][0], spec))
        n_target = sum(len(per_plot[n]["gt"]) for n in cf["test_plots"])
        allp = np.vstack([per_plot[n]["pred"] for n in cf["test_plots"] if len(per_plot[n]["pred"])])
        alls = np.concatenate([per_plot[n]["score"] for n in cf["test_plots"] if len(per_plot[n]["score"])])
        allg = np.vstack([per_plot[n]["gt"] for n in cf["test_plots"] if len(per_plot[n]["gt"])]) if n_target else np.zeros((0, 2))
        rnd = Y.random_baseline(cfg, per_plot, polys, radii)
        for r in radii:
            m = Y.match_distance(allp, alls, allg, r)
            edge = m["recall"] - rnd[r]["recall"]
            rows.append(dict(subset=label, field=field or "", pct=pct or "", radius_m=r,
                             n_target=n_target, n_pred=len(allp), tp=m["tp"],
                             precision=round(m["precision"], 4), recall=round(m["recall"], 4),
                             f1=round(m["f1"], 4), rand_recall=round(rnd[r]["recall"], 4),
                             rand_precision=round(rnd[r]["precision"], 4), edge_recall=round(edge, 4)))
            print(f"{label:11s} {n_target:>6} {len(allp):>6} | {r:>4} "
                  f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} "
                  f"{rnd[r]['recall']:>6.3f} {edge:>+6.3f}")
    od, _ = _od(cfg)
    with open(od / "size_eval.csv", "w", newline="") as f:
        fields = ["subset", "field", "pct", "radius_m", "n_target", "n_pred", "tp",
                  "precision", "recall", "f1", "rand_recall", "rand_precision", "edge_recall"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {od/'size_eval.csv'}")

    # predicted-vs-target at a few confidences (the count 'tell')
    print("\n  predicted count vs target count (over-prediction ratio):")
    print(f"    {'conf':>5} {'n_pred':>7} | " + " ".join(f"{s[0]:>10}" for s in subs))
    tgt = {}
    for label, field, pct, spec in subs:
        tgt[label] = sum(len(_subset_xy(cens[n][0], spec)) for n in cf["test_plots"])
    for cthr in [0.02, 0.03, 0.05]:
        ps = {n: pred_set(cfg, raw, n, cthr) for n in cf["test_plots"]}
        npred = sum(len(ps[n][0]) for n in cf["test_plots"])
        ratios = " ".join(f"{npred/max(1,tgt[s[0]]):>9.1f}x" for s in subs)
        print(f"    {cthr:>5} {npred:>7} | {ratios}")
    print("    (target counts: " + ", ".join(f"{s[0]}={tgt[s[0]]}" for s in subs) + ")")
    return rows


# ── stage 5: figures ─────────────────────────────────────────────────────────
def _fig_recall_curve(rd, rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, field, title in [(axes[0], "dbh", "DBH"), (axes[1], "totht", "TOTHT")]:
        for r, col in [(1.0, "#4575b4"), (1.5, "#7b3294")]:
            xs = [0] + PCTS
            mr = [next(x["recall"] for x in rows if x["subset"] == "all_alive" and x["radius_m"] == r)]
            er = [next(x["edge_recall"] for x in rows if x["subset"] == "all_alive" and x["radius_m"] == r)]
            for p in PCTS:
                lab = f"{field.upper() if field=='dbh' else 'TOTHT'}>p{p}"
                row = next((x for x in rows if x["subset"] == lab and x["radius_m"] == r), None)
                mr.append(row["recall"] if row else np.nan); er.append(row["edge_recall"] if row else np.nan)
            ax.plot(xs, mr, "-o", color=col, label=f"model recall @{r}m")
            ax.plot(xs, er, "--s", color=col, ms=4, alpha=0.7, label=f"edge over random @{r}m")
        ax.set_xlabel(f"{title} percentile kept (>pXX; 0 = all alive)")
        ax.set_ylabel("recall"); ax.set_ylim(-0.05, 1.0); ax.legend(fontsize=8)
        ax.set_title(f"s17 — recall vs {title} size cut")
        ax.axhline(0, color="#999", lw=0.6)
    fig.suptitle("s17 — does recall rise for larger trees? (label-visibility test)", fontsize=12)
    fig.tight_layout(); fig.savefig(rd/"recall_vs_size.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/'recall_vs_size.png'}")


def _fig_overlay(cfg, rd, cens, raw):
    import rasterio
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    name = cfg["bcnp_yolo"]["test_plots"][0]
    df = cens[name][0]
    alldbh = np.concatenate([d.dbh.dropna().values for d, _ in cens.values()])
    thr = np.percentile(alldbh, 90)
    big = df[df.dbh > thr]
    pw, _ = pred_set(cfg, raw, name, 0.02)
    with rasterio.open(Y.bc_dir(cfg) / f"{name}.tif") as src:
        tf = src.transform; step = max(1, int(round(max(src.width, src.height) / 1600)))
        img = src.read([1, 2, 3], out_shape=(3, src.height // step, src.width // step))
        H, W = img.shape[1], img.shape[2]

    def to_px(x, y):
        return (x - tf.c) / tf.a / step, (y - tf.f) / tf.e / step
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(np.transpose(img, (1, 2, 0)))
    if len(pw):
        p = [to_px(x, y) for x, y in pw]
        ax.scatter([q[0] for q in p], [q[1] for q in p], s=10, facecolors="none",
                   edgecolors="#d73027", linewidths=0.4, label=f"YOLO preds ({len(pw)})")
    g = [to_px(x, y) for x, y in big[["x", "y"]].values]
    ax.scatter([q[0] for q in g], [q[1] for q in g], s=40, c="#39ff14", marker="+",
               linewidths=1.1, label=f"census DBH>p90 ({len(big)}, >{thr:.0f} cm)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.legend(loc="upper right", fontsize=9, markerscale=1.5)
    ax.set_title(f"s17 — YOLO preds vs LARGE-tree-only census (DBH>p90) on {name}")
    fig.tight_layout(); fig.savefig(rd/f"overlay_bigtrees_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {rd/f'overlay_bigtrees_{name}.png'}")


def verdict(rows):
    print("\n========================= VERDICT =========================")
    def get(sub, r, k):
        row = next((x for x in rows if x["subset"] == sub and x["radius_m"] == r), None)
        return row[k] if row else None
    print("Recall & edge-over-random at 1.0 m as we keep only larger trees:")
    for sub in ["all_alive", "DBH>p50", "DBH>p75", "DBH>p90"]:
        print(f"  {sub:9s}: recall {get(sub,1.0,'recall')*100:5.1f}%  edge {get(sub,1.0,'edge_recall')*100:+5.1f} pts"
              f"  (target {get(sub,1.0,'n_target')})")
    r_all = get("all_alive", 1.0, "recall"); r_big = get("DBH>p90", 1.0, "recall")
    e_all = get("all_alive", 1.0, "edge_recall"); e_big = get("DBH>p90", 1.0, "edge_recall")
    rise = (r_big - r_all) * 100; erise = (e_big - e_all) * 100
    print(f"\nDBH recall change all->p90: {rise:+.1f} pts;  edge change: {erise:+.1f} pts.")
    if rise > 20 and e_big > 0.10:
        concl = "YES — recall rises sharply for big trees: the ceiling IS largely label visibility."
    elif rise > 8 or erise > 5:
        concl = "PARTLY — recall rises modestly for big trees; visibility explains SOME of the ceiling."
    else:
        concl = ("NO — recall barely moves with size; the model is not preferentially finding even "
                 "canopy-sized trees, so the ceiling is the model/density, not just visibility.")
    print(f"==> {concl}")
    print(f"peak RAM: {Y.peak_ram_gb():.1f} GB")
    print("NOTE: DBH 100% populated (reliable); TOTHT ~17% (recorded for a minority — read cautiously).")
    print("===========================================================")


def main():
    cfg = C.load_config(); np.random.seed(SEED)
    od, rd = _od(cfg)
    bcd = Y.bc_dir(cfg)
    census = pd.read_csv(bcd / "RP.Plot_census_data.2025.csv")
    polys = gpd.read_file(bcd / "tree_plots_polygons.geojson")
    cens = {n: census_full(cfg, n, census, polys) for n in cfg["bcnp_yolo"]["test_plots"]}

    distributions(cfg, rd, cens)
    raw = get_predictions(cfg, cens, census, polys)
    rows = evaluate(cfg, rd, cens, raw, polys)
    _fig_recall_curve(rd, rows)
    _fig_overlay(cfg, rd, cens, raw)
    verdict(rows)


if __name__ == "__main__":
    main()
