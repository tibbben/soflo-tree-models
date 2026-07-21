"""
crop_to_region.py — Crop the 5cm survey to the ground-truth polygon so the model is
only run where GT actually exists. Pixels outside the polygon are set to nodata.

Run from the project root:
    python scripts/crop_to_region.py

Reads : ./download/umgables_2025/umgables_2025_drone_survey_5cm.tif
        ./download/gt_region.geojson
Writes: ./download/umgables_2025/umgables_2025_drone_survey_5cm_gtregion.tif
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask

src_path = "./download/umgables_2025/umgables_2025_drone_survey_5cm.tif"
poly_path = "./download/gt_region.geojson"
dst_path = "./download/umgables_2025/umgables_2025_drone_survey_5cm_gtregion.tif"

# load the polygon, force it to the raster's CRS
region = gpd.read_file(poly_path)

with rasterio.open(src_path) as src:
    if region.crs != src.crs:
        region = region.to_crs(src.crs)
    shapes = [geom for geom in region.geometry]

    # crop=True trims the raster to the polygon's bounding box; pixels outside the
    # polygon itself are filled with nodata (0), so the model never sees them
    out_image, out_transform = mask(src, shapes, crop=True, nodata=0)

    profile = src.profile.copy()
    profile.update(height=out_image.shape[1],
                   width=out_image.shape[2],
                   transform=out_transform,
                   nodata=0)

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out_image)

print(f"Wrote {dst_path}")
print(f"  size: {out_image.shape[2]} x {out_image.shape[1]} px")
