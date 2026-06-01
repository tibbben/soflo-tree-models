"""
Step 03 — Prepare image tiles + DeepForest annotations.  [needs the .tif locally]

For each AOI: clip the orthomosaic to the AOI rectangle, convert that AOI's
world-coordinate boxes into pixel coordinates relative to the clip, write a
DeepForest annotation CSV, then tile the clip into training patches with
deepforest.preprocess.split_raster.

Tile size is specified on the ground (config tiles.patch_size_m) and converted to
pixels using the raster's GSD, so it is robust to whatever resolution your ortho is.

Output (per AOI): outputs/tiles/<aoi>/  with tile PNGs + <aoi>_annotations.csv
"""
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from PIL import Image
sys.path.append(str(Path(__file__).resolve().parent))
import common as C

# deepforest 2.x split_raster loads the full AOI clip through PIL; our clips are
# large (~180 Mpx) and trusted, so lift PIL's decompression-bomb ceiling.
Image.MAX_IMAGE_PIXELS = None


def main():
    cfg = C.load_config()
    from deepforest import preprocess  # imported here so other steps don't need DeepForest

    od = C.p(cfg, cfg["outputs_dir"])
    ortho_path = C.p(cfg, cfg["inputs"]["orthomosaic"])
    tiles_root = od / "tiles"; tiles_root.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ortho_path) as src:
        gsd = abs(src.transform.a)  # metres per pixel
        print(f"ortho: {src.width}x{src.height}, {src.count} bands, GSD ~ {gsd*100:.1f} cm/px, CRS {src.crs}")
    patch_px = int(round(cfg["tiles"]["patch_size_m"] / gsd))
    print(f"tile footprint {cfg['tiles']['patch_size_m']} m  ->  patch_size = {patch_px} px")

    for name in ("train", "test"):
        boxes = gpd.read_file(od / f"boxes_{name}.geojson").to_crs(cfg["crs"])
        aoi = gpd.read_file(od / f"aoi_{name}.geojson").geometry.iloc[0]

        clip_tif = tiles_root / f"{name}_clip.tif"
        with rasterio.open(ortho_path) as src:
            img, transform = mask(src, [aoi], crop=True)
            meta = src.meta.copy()
        # DeepForest expects 3-band RGB uint8; take first 3 bands
        img = img[:3]
        meta.update(height=img.shape[1], width=img.shape[2], transform=transform, count=img.shape[0])
        with rasterio.open(clip_tif, "w", **meta) as dst:
            dst.write(img)

        # world boxes -> pixel boxes relative to this clip
        recs = []
        for _, r in boxes.iterrows():
            xmin, ymin, xmax, ymax = C.world_box_to_pixel(r.geometry.bounds, transform)
            xmin = max(0, xmin); ymin = max(0, ymin)
            xmax = min(img.shape[2], xmax); ymax = min(img.shape[1], ymax)
            if xmax - xmin < 2 or ymax - ymin < 2:
                continue
            recs.append(dict(image_path=clip_tif.name, xmin=xmin, ymin=ymin,
                             xmax=xmax, ymax=ymax, label="Tree"))
        ann = pd.DataFrame(recs)
        ann_csv = tiles_root / f"{name}_clip_annotations.csv"
        ann.to_csv(ann_csv, index=False)
        print(f"{name}: {len(ann)} boxes in pixel space -> {ann_csv}")

        outdir = tiles_root / name; outdir.mkdir(exist_ok=True)
        tiled = preprocess.split_raster(   # deepforest 2.x: base_dir -> save_dir
            annotations_file=str(ann_csv),
            path_to_raster=str(clip_tif),
            save_dir=str(outdir),
            patch_size=patch_px,
            patch_overlap=cfg["tiles"]["patch_overlap"],
        )
        # drop tiles with too few boxes
        keep = tiled.groupby("image_path").filter(
            lambda g: len(g) >= cfg["tiles"]["min_boxes_per_tile"])
        keep.to_csv(outdir / f"{name}_tiles.csv", index=False)
        print(f"{name}: {keep.image_path.nunique()} tiles, {len(keep)} boxes -> {outdir}")

    _tiles_overview_figure(cfg, tiles_root)


def _tiles_overview_figure(cfg, tiles_root):
    """Render each AOI clip (downsampled) with its label boxes -> reports/tiles_overview.png."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    C.style()

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, name in zip(axes, ("train", "test")):
        clip_tif = tiles_root / f"{name}_clip.tif"
        ann_csv = tiles_root / f"{name}_clip_annotations.csv"
        if not clip_tif.exists():
            ax.axis("off"); continue
        with rasterio.open(clip_tif) as src:
            step = max(1, int(round(max(src.width, src.height) / 1500)))
            out_h, out_w = src.height // step, src.width // step
            img = src.read([1, 2, 3], out_shape=(3, out_h, out_w))
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ann = pd.read_csv(ann_csv) if ann_csv.exists() else pd.DataFrame()
        for _, r in ann.iterrows():
            ax.add_patch(Rectangle((r.xmin / step, r.ymin / step),
                                   (r.xmax - r.xmin) / step, (r.ymax - r.ymin) / step,
                                   fill=False, edgecolor="#1b7837", linewidth=0.4))
        ax.set_title(f"{name} AOI clip — {len(ann)} label boxes  (1:{step} downsample)")
        ax.axis("off")
    fig.suptitle("s03 — orthomosaic AOI clips with point-first label boxes", fontsize=12)
    fig.tight_layout()
    out = C.p(cfg, cfg["reports_dir"]) / "tiles_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
