"""
Step 06 — Predict trees over the whole test AOI and export to GeoJSON.  [local]

Runs windowed prediction over the test-AOI clip, converts pixel boxes back to
EPSG:32617, and writes outputs/predicted_trees.geojson — drop it straight into
QGIS on top of the ortho. Also writes a prediction-overlay figure and a count
comparison vs the inventory.
"""
import sys
from pathlib import Path
import geopandas as gpd
import rasterio
from PIL import Image
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

# deepforest 2.x predict_tile loads the full clip through PIL; lift the
# decompression-bomb ceiling for our large (~180 Mpx) trusted test clip.
Image.MAX_IMAGE_PIXELS = None


def _px_box(r):
    """Pixel box from a prediction row, whether columns or geometry carry it."""
    for c in ("xmin", "ymin", "xmax", "ymax"):
        if c not in r or r[c] is None:
            minx, miny, maxx, maxy = r.geometry.bounds
            return minx, miny, maxx, maxy
    return r["xmin"], r["ymin"], r["xmax"], r["ymax"]


def main():
    cfg = C.load_config()
    from deepforest import main as df_main

    od = C.p(cfg, cfg["outputs_dir"])
    clip = od / "tiles" / "test_clip.tif"

    m = df_main.deepforest.load_from_checkpoint(str(od / "model" / "treedetect_finetuned.ckpt"))
    m.config["score_thresh"] = cfg["train"]["score_thresh"]

    with rasterio.open(clip) as src:
        gsd = abs(src.transform.a)
        transform, crs = src.transform, src.crs
    patch_px = int(round(cfg["tiles"]["patch_size_m"] / gsd))

    # deepforest 2.x: predict_tile(path=..., patch_size=..., patch_overlap=...)
    pred = m.predict_tile(path=str(clip), patch_size=patch_px,
                          patch_overlap=cfg["tiles"]["patch_overlap"])
    if pred is None or len(pred) == 0:
        print("no predictions returned"); return
    pred = pred.reset_index(drop=True)
    print(f"raw predictions: {len(pred)}")

    px_boxes = [_px_box(r) for _, r in pred.iterrows()]
    geoms = [C.pixel_box_to_world(b[0], b[1], b[2], b[3], transform) for b in px_boxes]
    cols = [c for c in pred.columns if c != "geometry"]
    gdf = gpd.GeoDataFrame(pred[cols].assign(geometry=geoms), crs=crs).to_crs(cfg["crs"])
    out = od / "predicted_trees.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print(f"wrote {out}  ({len(gdf)} predicted trees)")

    gt = gpd.read_file(od / "boxes_test.geojson")
    print(f"inventory trees in test AOI: {len(gt)}   |   predicted: {len(gdf)}")

    _overlay_figure(cfg, clip, px_boxes, len(gt))


def _overlay_figure(cfg, clip, px_boxes, n_inventory):
    """Predicted boxes drawn on a downsampled view of the test-AOI clip."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    C.style()

    with rasterio.open(clip) as src:
        step = max(1, int(round(max(src.width, src.height) / 1800)))
        img = src.read([1, 2, 3], out_shape=(3, src.height // step, src.width // step))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(np.transpose(img, (1, 2, 0)))
    for minx, miny, maxx, maxy in px_boxes:
        ax.add_patch(Rectangle((minx / step, miny / step),
                               (maxx - minx) / step, (maxy - miny) / step,
                               fill=False, edgecolor="#d73027", linewidth=0.4))
    ax.set_title(f"s06 — predicted trees on test-AOI clip\n"
                 f"{len(px_boxes)} predicted vs {n_inventory} inventory boxes  (1:{step} downsample)")
    ax.axis("off")
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "predict_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
