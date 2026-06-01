"""Shared utilities for the treedetect pipeline."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    return cfg


def p(cfg, rel):
    """Resolve a repo-relative path from config to an absolute Path."""
    return cfg["_root"] / rel


def aoi_bounds(aoi_cfg, data_bounds):
    """Return [minx, miny, maxx, maxy] for an AOI given the data bounding box.

    Explicit `bounds` win; otherwise carve a corner using bbox_fraction.
    """
    if aoi_cfg.get("bounds"):
        return list(aoi_cfg["bounds"])
    xmin, ymin, xmax, ymax = data_bounds
    fx, fy = aoi_cfg["bbox_fraction"]
    w, h = (xmax - xmin) * fx, (ymax - ymin) * fy
    corner = aoi_cfg["corner"]
    if corner == "bottom_left":
        return [xmin, ymin, xmin + w, ymin + h]
    if corner == "top_right":
        return [xmax - w, ymax - h, xmax, ymax]
    if corner == "bottom_right":
        return [xmax - w, ymin, xmax, ymin + h]
    if corner == "top_left":
        return [xmin, ymax - h, xmin + w, ymax]
    raise ValueError(f"unknown corner {corner}")


# ── matplotlib house style for research figures ────────────────────────────
def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "font.size": 10, "axes.titlesize": 11, "savefig.dpi": 150,
        "axes.grid": False,
    })

SRC_COLORS = {
    "polygon":       "#1b7837",   # reliable crown extent
    "merged_fixed":  "#f1a340",   # inside a merged (fused) polygon
    "nomatch_fixed": "#d73027",   # missed by the prior model
}


def world_box_to_pixel(bounds, transform):
    """Convert a world-coordinate (minx,miny,maxx,maxy) box to pixel
    (xmin,ymin,xmax,ymax) using a rasterio affine transform. Row 0 is the top."""
    from rasterio.transform import rowcol
    minx, miny, maxx, maxy = bounds
    r_top, c_left = rowcol(transform, minx, maxy)
    r_bot, c_right = rowcol(transform, maxx, miny)
    xmin, xmax = sorted((c_left, c_right))
    ymin, ymax = sorted((r_top, r_bot))
    return xmin, ymin, xmax, ymax


def pixel_box_to_world(xmin, ymin, xmax, ymax, transform):
    """Inverse of the above: pixel box -> shapely world-coordinate box."""
    from rasterio.transform import xy
    from shapely.geometry import box
    x0, y0 = xy(transform, ymin, xmin)   # top-left
    x1, y1 = xy(transform, ymax, xmax)   # bottom-right
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
