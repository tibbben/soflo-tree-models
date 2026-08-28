"""
Step 07 — Build a SPECIES-LABELED dataset (labels + visuals only; no training).

Prep for a future classification stage. We take the EXISTING point-first
detection boxes from s01 (outputs/detection_boxes.geojson), join the botanical
family onto each box by inventory point id (pt_id), and collapse it into a small
set of target classes (palm / pine / cypress / other) via config `species.species_map`.
Box geometry is left byte-for-byte identical to the detection labels — we only
add a `species_class` column.

Outputs (outputs/species/, reports/species/):
  - species_boxes.geojson   world-coord boxes + species_class  (drop into QGIS / retile)
  - species_boxes.csv       flat label table: pt_id, src, family, species, species_class,
                            cx, cy, minx..maxy  — class is a COLUMN, so it feeds either
                            a multi-class detector (after tiling) OR a crop classifier.
  - class_distribution.csv  per-class counts + share, incl. missing/unknown
  - reports/species/species_class_bar.png      count per species_class
  - reports/species/species_points_map.png     campus points colored by class
  - reports/species/species_sample_tile.png    one tile, boxes colored by class over imagery

Run:  SOFLO_DATA_ROOT=/path/to/soflo_data .venv/bin/python src/s07_species_labels.py
The campus map + bar chart need no imagery; the sample tile reads a small window
from the orthomosaic (skipped gracefully if the ortho is not reachable).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

# Stable colors per target class (+ unknown) used across all three figures.
CLASS_COLORS = {
    "palm":    "#1b7837",   # green  — Arecaceae
    "pine":    "#4575b4",   # blue   — Pinaceae
    "cypress": "#7b3294",   # purple — Cupressaceae
    "other":   "#999999",   # grey   — every other family
    "unknown": "#d9d9d9",   # light  — no usable family value
}


def _usable(s):
    """A taxonomy string is usable if non-null, non-blank, not a 'none' sentinel."""
    t = s.astype(str).str.strip()
    return s.notna() & (t != "") & (t.str.lower() != "none") & (t.str.lower() != "nan")


def classify(family, cfg):
    """family value -> target class, using config species.species_map."""
    sp = cfg["species"]
    smap = sp["species_map"]
    miss = sp["missing_class"]
    if family is None:
        return miss
    f = str(family).strip()
    if f == "" or f.lower() in ("none", "nan"):
        return miss
    return smap.get(f, "other")


def main():
    cfg = C.load_config()
    L, S = cfg["labels"], cfg["species"]
    src_field = S["source_field"]

    # 1) the existing s01 detection boxes — geometry we must preserve exactly.
    od = C.p(cfg, cfg["outputs_dir"])
    boxes = gpd.read_file(od / "detection_boxes.geojson")
    print(f"detection boxes (from s01): {len(boxes)}")

    # 2) rebuild the SAME confirmed-tree frame s01 used, so pt_id is the join key
    #    (s01: trees = pts[flag=='Yes'].reset_index(drop=True); pt_id = that index).
    pts = gpd.read_file(C.input_path(cfg, "points")).to_crs(cfg["crs"])
    trees = pts[pts[L["tree_flag_field"]] == L["tree_flag_value"]].copy().reset_index(drop=True)
    trees["pt_id"] = trees.index
    if len(trees) != len(boxes):
        print(f"WARNING: confirmed-tree count {len(trees)} != boxes {len(boxes)} "
              f"(s01 may have been rebuilt) — joining on pt_id regardless")

    fam_by_id = trees.set_index("pt_id")[src_field]
    sp_by_id = trees.set_index("pt_id")[S.get("fallback_field", L["species_field"])]

    # 3) join family + classify (geometry untouched).
    boxes["tax_family"] = boxes["pt_id"].map(fam_by_id)
    if "species" not in boxes.columns:
        boxes["species"] = boxes["pt_id"].map(sp_by_id)
    boxes["species_class"] = [classify(f, cfg) for f in boxes["tax_family"]]

    # ── report: per-class counts + missing share ──────────────────────────
    order = list(S["target_classes"]) + [S["missing_class"]]
    counts = boxes["species_class"].value_counts()
    n = len(boxes)
    n_missing = int(counts.get(S["missing_class"], 0))
    print("\n=== species_class distribution (all boxes) ===")
    dist_rows = []
    for c in order:
        v = int(counts.get(c, 0))
        print(f"  {c:9s} {v:5d}  ({v/n*100:5.1f}%)")
        dist_rows.append({"species_class": c, "n_boxes": v, "share_pct": round(v / n * 100, 2)})
    print(f"  {'TOTAL':9s} {n:5d}")
    print(f"\nusable species (family present): {n - n_missing}/{n} "
          f"({(n-n_missing)/n*100:.1f}%)   missing/unknown: {n_missing} ({n_missing/n*100:.1f}%)")

    # class-imbalance flag (among trainable classes only)
    trainable = {c: int(counts.get(c, 0)) for c in S["target_classes"]}
    big = max(trainable, key=trainable.get); small = min(trainable, key=trainable.get)
    if trainable[small] > 0:
        ratio = trainable[big] / trainable[small]
        print(f"\nclass imbalance: '{big}' ({trainable[big]}) : '{small}' ({trainable[small]}) "
              f"= {ratio:.0f}:1")
    rare = [c for c, v in trainable.items() if v < 0.05 * (n - n_missing)]
    if rare:
        print(f"IMBALANCE FLAG: rare classes (<5% of usable): {', '.join(rare)}")

    # 4) write label set ──────────────────────────────────────────────────
    sd = od / "species"; sd.mkdir(parents=True, exist_ok=True)
    gj = sd / "species_boxes.geojson"
    boxes.to_file(gj, driver="GeoJSON")
    print(f"\nwrote {gj}  ({len(boxes)} boxes)")

    csv_cols = ["pt_id", "src", "tax_family", "species", "species_class",
                "cx", "cy", "minx", "miny", "maxx", "maxy"]
    flat = pd.DataFrame(boxes.drop(columns="geometry"))[csv_cols]
    csv = sd / "species_boxes.csv"
    flat.to_csv(csv, index=False)
    print(f"wrote {csv}")

    dist = sd / "class_distribution.csv"
    pd.DataFrame(dist_rows).to_csv(dist, index=False)
    print(f"wrote {dist}")

    # 5) visuals ───────────────────────────────────────────────────────────
    _bar_figure(cfg, counts, order)
    _points_map(cfg, boxes)
    _sample_tile(cfg, boxes)


def _bar_figure(cfg, counts, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.style()
    vals = [int(counts.get(c, 0)) for c in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(order)), vals,
                  color=[CLASS_COLORS.get(c, "#999999") for c in order], width=.64)
    tot = sum(vals)
    for b, v in zip(bars, vals):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v + tot * 0.01,
                    f"{v}\n{v/tot*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("labeled tree boxes"); ax.set_ylim(0, max(vals) * 1.16)
    ax.set_title("s07 — tree count per species_class (UM Gables inventory)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "species" / "species_class_bar.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _points_map(cfg, boxes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C.style()
    fig, ax = plt.subplots(figsize=(9, 9))
    cx, cy = boxes["cx"].values, boxes["cy"].values
    cls = boxes["species_class"].values
    # draw rare/legible classes on top of the dense palm/other layers
    z = {"other": 0, "unknown": 1, "palm": 2, "pine": 3, "cypress": 4}
    for c in sorted(set(cls), key=lambda k: z.get(k, 0)):
        m = cls == c
        ax.scatter(cx[m], cy[m], s=(5 if c in ("other", "unknown", "palm") else 26),
                   c=CLASS_COLORS.get(c, "#999999"), alpha=.7, linewidths=0, label=c)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("s07 — inventory trees colored by species_class (EPSG:32617)")
    handles = [Line2D([0], [0], marker="o", ls="", c=CLASS_COLORS[c],
               label=f"{c} ({int((cls==c).sum())})") for c in z if (cls == c).any()]
    ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "species" / "species_points_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def _sample_tile(cfg, boxes):
    """One ~70 m window over the imagery, boxes colored by species_class.
    Centered to include a rarer class (cypress, else pine) when possible."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        import rasterio
        from rasterio.windows import from_bounds
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        C.style()

        win_m = 70.0
        # pick a center: prefer cypress, then pine, then the densest palm area
        for pref in ("cypress", "pine"):
            sub = boxes[boxes["species_class"] == pref]
            if len(sub):
                cx0, cy0 = float(sub["cx"].median()), float(sub["cy"].median())
                focus = pref
                break
        else:
            cx0, cy0 = float(boxes["cx"].median()), float(boxes["cy"].median())
            focus = "palm/other"
        minx, maxx = cx0 - win_m / 2, cx0 + win_m / 2
        miny, maxy = cy0 - win_m / 2, cy0 + win_m / 2

        ortho = C.input_path(cfg, "orthomosaic")
        with rasterio.open(ortho) as srcr:
            win = from_bounds(minx, miny, maxx, maxy, srcr.transform)
            img = srcr.read([1, 2, 3], window=win)
            wt = srcr.window_transform(win)
        H, W = img.shape[1], img.shape[2]
        if H == 0 or W == 0:
            print("sample tile skipped: empty window"); return

        # boxes whose center is inside the window
        m = ((boxes["cx"] >= minx) & (boxes["cx"] <= maxx)
             & (boxes["cy"] >= miny) & (boxes["cy"] <= maxy))
        sub = boxes[m]

        def to_px(x, y):
            col = (x - wt.c) / wt.a
            row = (y - wt.f) / wt.e
            return col, row

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(np.transpose(img, (1, 2, 0)))
        present = []
        for _, r in sub.iterrows():
            x0, y0 = to_px(r["minx"], r["maxy"])   # top-left
            x1, y1 = to_px(r["maxx"], r["miny"])   # bottom-right
            col = CLASS_COLORS.get(r["species_class"], "#999999")
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=col, linewidth=1.6))
            present.append(r["species_class"])
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(f"s07 — sample tile ({win_m:.0f} m, focus={focus})  "
                     f"{len(sub)} boxes colored by species_class")
        order = [c for c in CLASS_COLORS if c in set(present)]
        ax.legend(handles=[Line2D([0], [0], color=CLASS_COLORS[c], lw=3,
                  label=f"{c} ({present.count(c)})") for c in order],
                  loc="upper right", fontsize=9, frameon=True)
        fig.tight_layout()
        out = C.p(cfg, cfg["reports_dir"]) / "species" / "species_sample_tile.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")
    except Exception as e:
        print("sample tile skipped:", e)


if __name__ == "__main__":
    main()
