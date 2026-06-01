"""
Step 02 — Define train/test AOIs and split the labels.

Reads the AOI definitions from config (corner fractions, or explicit bounds you
read off QGIS), writes each AOI rectangle as a GeoJSON you can load in QGIS, and
splits detection_boxes.geojson into train/test label sets.

Output: outputs/aoi_train.geojson, outputs/aoi_test.geojson,
        outputs/boxes_train.geojson, outputs/boxes_test.geojson
        reports/aoi_split.png
"""
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import common as C


def main():
    cfg = C.load_config()
    od = C.p(cfg, cfg["outputs_dir"])
    boxes = gpd.read_file(od / "detection_boxes.geojson")
    data_bounds = boxes.total_bounds  # minx, miny, maxx, maxy

    split = {}
    for name in ("train", "test"):
        b = C.aoi_bounds(cfg["aois"][name], data_bounds)
        aoi = gpd.GeoDataFrame({"aoi": [name]}, geometry=[box(*b)], crs=cfg["crs"])
        aoi.to_file(od / f"aoi_{name}.geojson", driver="GeoJSON")
        sel = boxes[boxes.geometry.centroid.within(box(*b))].copy()
        sel.to_file(od / f"boxes_{name}.geojson", driver="GeoJSON")
        split[name] = (b, sel)
        print(f"{name}: {len(sel)} trees   bounds={[round(v,1) for v in b]}")

    _split_figure(cfg, boxes, split)


def _split_figure(cfg, boxes, split):
    C.style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5)); ax.set_aspect("equal")
    cx, cy = boxes["cx"].values, boxes["cy"].values
    ax.scatter(cx, cy, s=2, c="#cccccc", alpha=.5, linewidths=0)
    cols = {"train": "#2166ac", "test": "#b2182b"}
    for name, (b, sel) in split.items():
        ax.scatter(sel["cx"], sel["cy"], s=3, c=cols[name], alpha=.7, linewidths=0)
        ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=cols[name], lw=2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Train / test AOI split")
    ax.legend(handles=[Patch(color=cols[n], label=f"{n} ({len(s[1])})") for n, s in split.items()],
              loc="lower center", bbox_to_anchor=(.5, -.08), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "aoi_split.png"
    fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


if __name__ == "__main__":
    main()
