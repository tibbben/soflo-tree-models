"""
Step 01 — Build detection labels.

Strategy (see README): the curated inventory POINTS are ground truth for tree
location; the prior model's POLYGONS are unreliable (low recall, merged crowns),
so we only borrow a polygon's extent when it cleanly contains exactly one tree.

Output: outputs/detection_boxes.geojson — one box per confirmed tree, tagged with
`src` (polygon | merged_fixed | nomatch_fixed) so label provenance is auditable.
Also writes a QA figure to reports/label_qa.png.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import common as C


def main():
    cfg = C.load_config()
    L = cfg["labels"]

    pts = gpd.read_file(C.p(cfg, cfg["inputs"]["points"])).to_crs(cfg["crs"])
    seg = gpd.read_file(C.p(cfg, cfg["inputs"]["segmentation"])).to_crs(cfg["crs"])
    seg = seg.reset_index(drop=True)
    seg["seg_id"] = seg.index

    trees = pts[pts[L["tree_flag_field"]] == L["tree_flag_value"]].copy().reset_index(drop=True)
    trees["pt_id"] = trees.index
    print(f"confirmed trees: {len(trees)} / {len(pts)} inventory points")

    j = gpd.sjoin(trees, seg[["seg_id", "geometry"]], how="left", predicate="within")
    j = j.drop_duplicates("pt_id")          # a point inside overlapping polys -> keep one
    cnt = j.dropna(subset=["seg_id"]).groupby("seg_id").size()
    clean = set(cnt[cnt == 1].index)
    merged = set(cnt[cnt > 1].index)

    # fallback box size = median clean-crown dimension, unless overridden in config
    sb = seg.geometry.bounds
    seg["wm"], seg["hm"] = sb.maxx - sb.minx, sb.maxy - sb.miny
    cs = seg.loc[seg.seg_id.isin(clean), ["wm", "hm"]]
    fb = L["fallback_box_m"] or float(np.median(np.r_[cs.wm.values, cs.hm.values]))
    half = fb / 2.0
    print(f"fallback box size: {fb:.2f} m")

    seg_geom = seg.set_index("seg_id").geometry
    rows = []
    for _, r in j.iterrows():
        sid, x, y = r["seg_id"], r.geometry.x, r.geometry.y
        if pd.notna(sid) and sid in clean:
            mnx, mny, mxx, mxy = seg_geom.loc[sid].bounds
            src = "polygon"
        else:
            mnx, mny, mxx, mxy = x - half, y - half, x + half, y + half
            src = "merged_fixed" if (pd.notna(sid) and sid in merged) else "nomatch_fixed"
        rows.append(dict(pt_id=r["pt_id"], src=src, species=r.get(L["species_field"]),
                         cx=x, cy=y, minx=mnx, miny=mny, maxx=mxx, maxy=mxy,
                         geometry=box(mnx, mny, mxx, mxy)))

    boxes = gpd.GeoDataFrame(rows, crs=cfg["crs"])
    out = C.p(cfg, cfg["outputs_dir"]) / "detection_boxes.geojson"
    boxes.to_file(out, driver="GeoJSON")
    print("box source counts:\n" + boxes["src"].value_counts().to_string())
    print(f"wrote {out}  ({len(boxes)} boxes)")

    _qa_figure(cfg, boxes)


def _qa_figure(cfg, boxes):
    C.style()
    cx, cy = boxes["cx"].values, boxes["cy"].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for s, col in C.SRC_COLORS.items():
        m = boxes["src"].values == s
        axes[0].scatter(cx[m], cy[m], s=3, c=col, alpha=.6, linewidths=0)
    axes[0].set_aspect("equal"); axes[0].set_xticks([]); axes[0].set_yticks([])
    n_miss = int((boxes["src"] == "nomatch_fixed").sum())
    axes[0].set_title(f"Tree labels by source\n{n_miss} trees missed by prior model")
    axes[0].legend(handles=[Patch(color=c, label=s) for s, c in C.SRC_COLORS.items()],
                   loc="lower center", bbox_to_anchor=(.5, -.18), fontsize=8, frameon=False)

    counts = boxes["src"].value_counts()
    order = ["polygon", "merged_fixed", "nomatch_fixed"]
    vals = [int(counts.get(s, 0)) for s in order]
    bars = axes[1].bar(range(3), vals, color=[C.SRC_COLORS[s] for s in order], width=.62)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 60, f"{v}\n{v/sum(vals)*100:.0f}%",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(["reliable\ncrown polygon", "merged\nwith neighbours", "missed by\nprior model"], fontsize=8)
    axes[1].set_ylabel("confirmed trees"); axes[1].set_ylim(0, max(vals) * 1.18)
    axes[1].set_title("Where each label comes from")
    for sp in ("top", "right"):
        axes[1].spines[sp].set_visible(False)

    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "label_qa.png"
    fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


if __name__ == "__main__":
    main()
