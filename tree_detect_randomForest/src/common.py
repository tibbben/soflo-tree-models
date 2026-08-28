"""Shared utilities for the treedetect pipeline."""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Load a .env next to config.yaml (if present) so SOFLO_DATA_ROOT can live there.
# python-dotenv is optional at import time; it is listed in requirements.txt.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def swap_enabled(cfg):
    """Whether to exchange the train/test AOI definitions (diagnostic).

    Env ``TD_SWAP_AOIS`` (1/true/yes/on) overrides the config ``swap_aois`` flag.
    """
    v = os.environ.get("TD_SWAP_AOIS")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(cfg.get("swap_aois", False))


def load_config():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    # Optional output redirect so a tagged run (e.g. the swap diagnostic) writes
    # to a subfolder of outputs/ and reports/ instead of overwriting the main run.
    # Explicit TD_OUT_SUBDIR wins; otherwise a swapped run defaults to "swap".
    sub = os.environ.get("TD_OUT_SUBDIR")
    if sub is None:
        sub = "swap" if swap_enabled(cfg) else ""
    sub = sub.strip()
    if sub:
        cfg["outputs_dir"] = str(Path(cfg["outputs_dir"]) / sub)
        cfg["reports_dir"] = str(Path(cfg["reports_dir"]) / sub)
    return cfg


def p(cfg, rel):
    """Resolve a repo-relative path (outputs / figures) to an absolute Path."""
    return cfg["_root"] / rel


def data_root(cfg):
    """Resolve the external data directory holding the inputs (not in the repo).

    Order: env ``SOFLO_DATA_ROOT`` > ``data_root`` in config.yaml > clear error.
    """
    root = os.environ.get("SOFLO_DATA_ROOT") or (cfg.get("data_root") if cfg else None)
    if not root:
        raise RuntimeError(
            "treedetect: no data root configured. The input data is NOT stored in "
            "the repo — point the pipeline at your local copy by either:\n"
            "  1. exporting  SOFLO_DATA_ROOT=/path/to/soflo_data , or\n"
            "  2. copying  .env.example -> .env  and setting SOFLO_DATA_ROOT there, or\n"
            "  3. setting  data_root:  in config.yaml.\n"
            "See .env.example for the expected data folder layout."
        )
    root = Path(root).expanduser()
    if not root.exists():
        raise RuntimeError(
            f"treedetect: data root does not exist: {root}\n"
            "Set SOFLO_DATA_ROOT to your local soflo_data directory (see .env.example)."
        )
    return root


def input_path(cfg, key):
    """Resolve an input file (orthomosaic / geojsons / metadata) under DATA_ROOT.

    The path is ``DATA_ROOT / cfg['inputs'][key]`` (config values are relative).
    """
    rel = cfg["inputs"][key]
    return data_root(cfg) / rel


def best_checkpoint(model_dir):
    """Resolve the checkpoint to evaluate: prefer best-by-val-recall, then the
    last epoch, then the legacy final checkpoint, else the newest .ckpt."""
    model_dir = Path(model_dir)
    for name in ("treedetect_best_recall.ckpt", "last.ckpt", "treedetect_finetuned.ckpt"):
        pth = model_dir / name
        if pth.exists():
            return pth
    ckpts = sorted(model_dir.glob("*.ckpt"), key=lambda q: q.stat().st_mtime)
    return ckpts[-1] if ckpts else model_dir / "treedetect_finetuned.ckpt"


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
