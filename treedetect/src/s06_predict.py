"""
Step 06 — Predict trees over the whole test AOI and export to GeoJSON.  [local]

Runs windowed prediction over the test-AOI clip, converts pixel boxes back to
EPSG:32617, and writes outputs/predicted_trees.geojson — drop it straight into
QGIS on top of the ortho. Also writes a count comparison vs the inventory.
"""
import sys
from pathlib import Path
import geopandas as gpd
import rasterio
sys.path.append(str(Path(__file__).resolve().parent))
import common as C


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

    pred = m.predict_tile(raster_path=str(clip), patch_size=patch_px,
                          patch_overlap=cfg["tiles"]["patch_overlap"], return_plot=False)
    print(f"raw predictions: {len(pred)}")

    geoms = [C.pixel_box_to_world(r.xmin, r.ymin, r.xmax, r.ymax, transform)
             for _, r in pred.iterrows()]
    gdf = gpd.GeoDataFrame(pred.assign(geometry=geoms), crs=crs).to_crs(cfg["crs"])
    out = od / "predicted_trees.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print(f"wrote {out}  ({len(gdf)} predicted trees)")

    gt = gpd.read_file(od / "boxes_test.geojson")
    print(f"inventory trees in test AOI: {len(gt)}   |   predicted: {len(gdf)}")


if __name__ == "__main__":
    main()
