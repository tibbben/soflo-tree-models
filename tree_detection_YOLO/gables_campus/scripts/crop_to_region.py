"""
crop_to_region.py — Crop the survey to the evaluation polygon so detection and
benchmarking only happen where ground truth actually exists. Pixels inside the
polygon's bounding box but outside the polygon itself are set to nodata (0).

The detector skips tiles whose mean pixel value falls below a threshold, so those
zeroed areas produce no detections at all and cannot be counted as false positives.

MEMORY NOTE: this reads only the polygon's bounding-box window, not the whole raster.
A naive rasterio.mask.mask() call over the full survey loads every pixel into memory
and can be killed on a memory-limited machine.

Run from the project root:
    python scripts/crop_to_region.py

Reads : ./download/umgables_2025/umgables_2025_drone_survey_5cm.tif
        ./download/gt_region.geojson
Writes: ./download/umgables_2025/umgables_2025_drone_survey_5cm_gtregion.tif

To crop a different survey (e.g. a downsampled one), edit the three paths below. This
script is deliberately not config-driven; it is only run when a new source raster is
introduced.

Written by Ahsan and Claude.
"""

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask

src_path = "./download/umgables_2025/umgables_2025_drone_survey_5cm.tif"
poly_path = "./download/gt_region.geojson"
dst_path = "./download/umgables_2025/umgables_2025_drone_survey_5cm_gtregion.tif"

region = gpd.read_file(poly_path)

with rasterio.open(src_path) as src:
    # force the polygon into the raster's CRS before using its coordinates
    r = region.to_crs(src.crs) if region.crs != src.crs else region
    poly = r.union_all() if hasattr(r, "union_all") else r.unary_union

    # windowed read of the polygon's bounding box only — this is the memory-safe part
    minx, miny, maxx, maxy = poly.bounds
    win = from_bounds(minx, miny, maxx, maxy, src.transform)
    win = win.round_offsets().round_lengths()
    data = src.read(window=win)

    # zero everything inside the bounding box but outside the polygon itself
    win_transform = src.window_transform(win)
    outside = geometry_mask([poly], out_shape=(data.shape[1], data.shape[2]),
                            transform=win_transform, invert=False)
    data[:, outside] = 0

    profile = src.profile.copy()
    profile.update(height=data.shape[1], width=data.shape[2],
                   transform=win_transform, nodata=0)

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)

print(f"Wrote {dst_path}")
print(f"  size: {data.shape[2]} x {data.shape[1]} px")
