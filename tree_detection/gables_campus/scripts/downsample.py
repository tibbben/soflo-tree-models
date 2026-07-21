"""
downsample.py — Downsample the 5cm survey to a coarser resolution GeoTIFF.

Reads the 5cm drone survey and writes a coarser copy at the requested resolution,
using area-averaging resampling (proper anti-aliased downsampling). CRS, band count,
and dtype are preserved; only pixel size / dimensions change.

Run from the project root:
    python scripts/downsample.py [target_cm]     # default 7.5

Output: ./download/umgables_2025/umgables_2025_drone_survey_<target>cm.tif
"""

import sys
import rasterio
from rasterio.enums import Resampling

# Target resolution in cm from the command line (default 7.5).
target_cm = float(sys.argv[1]) if len(sys.argv) > 1 else 7.5
target_res = target_cm / 100.0   # cm -> meters per pixel

src_path = "./download/umgables_2025/umgables_2025_drone_survey_5cm.tif"
# Filename tag: whole numbers drop the trailing ".0" (20.0 -> "20"), decimals use "_"
# in place of the dot (7.5 -> "7_5"), so the extension is the only "." in the name.
if target_cm == int(target_cm):
    tag = str(int(target_cm))            # 20.0 -> "20", 10.0 -> "10"
else:
    tag = str(target_cm).replace(".", "_")   # 7.5 -> "7_5"
dst_path = f"./download/umgables_2025/umgables_2025_drone_survey_{tag}cm.tif"

with rasterio.open(src_path) as src:
    # New dimensions to hit the target pixel size.
    scale = src.res[0] / target_res              # 0.05 / 0.075 = 0.6667
    new_w = int(src.width * scale)
    new_h = int(src.height * scale)
    print(f"{src.width}x{src.height} @ {src.res[0]}m/px -> {new_w}x{new_h} @ {target_res}m/px")

    # Read the whole raster resampled to the new shape (GDAL averages down blockwise,
    # so it does not load the full-resolution array into memory).
    data = src.read(out_shape=(src.count, new_h, new_w), resampling=Resampling.average)

    # Rescale the affine transform to match the new pixel size.
    transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2]),
    )

    profile = src.profile.copy()
    profile.update(width=new_w, height=new_h, transform=transform)

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)

print(f"Wrote {dst_path}")