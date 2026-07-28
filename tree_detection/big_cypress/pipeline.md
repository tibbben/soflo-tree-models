# Pipeline

How to run tree detection on the Big Cypress plot clips.

This is a **detection-only** pipeline. There is no training and no benchmarking, because
the site has no usable ground truth yet — see `progress_summary.md`. Two pretrained models
are applied to the imagery and their outputs compared visually.

---

## 1. Overview

```
plot clips ──┬──► detect_yolo.py       ──► points ──► QGIS
             │      (campus champion, 5cm)
             │
             └──► detect_deepforest.py ──► points ──► QGIS
                    (release weights, 10cm)
```

Each detector reads its own YAML config. The two configs differ in chip geometry, which is
deliberate — see §5.

---

## 2. Requirements

**Two separate environments.** The DeepForest detector cannot run in the YOLO environment.

YOLO detector:

```bash
conda create -n treedetection python=3.10 -y
conda activate treedetection
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "ultralytics>=8.4.84" rasterio geopandas shapely pyyaml scipy pillow
```

The ultralytics version matters — earlier releases do not recognise the YOLO26 weight
strings. Install torch first from the CUDA index so a CPU build is not pulled in.

DeepForest detector:

```bash
conda create -n deepforest python=3.10 -y
conda activate deepforest
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "deepforest==1.5.2" "albumentations<2.0" rasterio geopandas pyyaml scipy
```

Install torch **first** from the CUDA index — letting the DeepForest resolver pull it
produces a CPU build. The pins matter: DeepForest 2.x rewrote the config API, and
albumentations 2.0 removed an import that 1.5.2 depends on.

Verify:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "from deepforest import main; m=main.deepforest(); m.load_model('weecology/deepforest-tree'); print('weights ok')"
```

---

## 3. Directory layout

```
big_cypress/
├── configs/
│   ├── bigcypress_plots.yaml             # YOLO geometry (5cm effective)
│   └── bigcypress_plots_deepforest.yaml  # DeepForest geometry (10cm effective)
│
├── scripts/
│   ├── config.py                 # shared config loader; imported, not run directly
│   ├── detect_yolo.py            # YOLO detection across all plot clips
│   └── detect_deepforest.py      # DeepForest release-model detection
│
├── yolo_champion_0.15conf.qgz         # QGIS projects, one per model and threshold
├── yolo_champion_0.20conf.qgz
├── yolo_champion_0.25conf.qgz
├── deepforest_pretrained_0.05conf.qgz
├── deepforest_pretrained_0.10conf.qgz
├── deepforest_pretrained_0.15conf.qgz
│
├── README.md
├── pipeline.md
└── progress_summary.md
```

Generated at runtime and gitignored:

```
big_cypress/
├── download/
│   └── ortho_clipped/            # 51 per-plot .tif clips
└── output/
    ├── yolo/<plot>.geojson
    ├── yolo_all_plots.geojson
    ├── deepforest/<plot>.geojson
    └── deepforest_all_plots.geojson
```

Place the download folder from the shared Big Cypress Box folder so that the clips sit at
`./download/ortho_clipped/`.

---

## 4. Input data

| Property | Value |
|---|---|
| Files | 51 per-plot orthomosaic clips |
| Ground sample distance | 1.69 cm |
| Clip size | ~9466 × 9466 px = 160 m square (2.56 ha) |
| Bands | 4 (uint8) — band 4 is alpha, constant 255, **not** NIR |
| CRS | EPSG:32617 |

The detectors read the first three bands only. Non-uint8 imagery is scaled rather than
rejected, so other surveys can be run through the same scripts.

---

## 5. The two geometries

Chips are cut by **ground distance**, then resampled to a fixed pixel size. This means the
effective resolution presented to a model is set by the config, and the source imagery is
downsampled on the fly — no intermediate files are created.

| | Chip | Ground | Source px read | Effective GSD |
|---|---|---|---|---|
| YOLO | 640 px | 32 m | 1893 | 5 cm |
| DeepForest | 400 px | 40 m | 2367 | 10 cm |

The campus champion was trained on 5 cm imagery in 640 px / 32 m chips. The DeepForest
release model was trained on NEON imagery at ~10 cm in 400 × 400 px patches. Each is fed
the geometry it was trained on.

This matters more than it would with fine-tuning. Neither model is being adapted to this
site, so a scale mismatch would go straight through uncorrected — crowns would appear at
the wrong pixel size and the model would have no opportunity to learn around it.

The consequence is that the two models do **not** see identical inputs, so this is not a
controlled architecture comparison in the way the campus experiments were.

Downsampling uses **area averaging**, not bilinear interpolation. Bilinear samples sparse
points when shrinking an image, which aliases and manufactures spurious edge texture across
continuous canopy. Averaging is the anti-aliased choice whenever the output is smaller than
the source window, which is always the case here.

---

## 6. Configuration

```yaml
name: bigcypress_plots

data:
  plot_dir: ./download/ortho_clipped   # directory of per-plot .tif clips
  out_size: 640          # every chip is emitted at this pixel size
  ground_m: 32.0         # every chip covers this much ground
  radius_m: 5.0          # reference crown radius — printed summary only, see below
  overlap_m: 5.0         # tile overlap in metres, resolution-independent

detect:
  imgsz: 1280
  nms_iou: 0.5
```

`radius_m` **does not affect detection.** It only feeds the printed summary line. The size
of the boxes a model predicts is fixed in its trained weights; it is not a parameter here.
Changing crown box size at this site would mean retraining elsewhere and transferring, or
waiting for local labels.

If detections look badly scaled — clusters of tiny boxes on one crown, or one box spanning
several trees — `ground_m` is the lever. Lower it to zoom in, raise it to zoom out.

---

## 7. Running

YOLO, using the campus champion weights:

```bash
conda activate treedetection
python scripts/detect_yolo.py ./configs/bigcypress_plots.yaml <path/to/best.pt> 0.05
```

DeepForest, using the release weights (downloaded automatically on first use):

```bash
conda activate deepforest
python scripts/detect_deepforest.py ./configs/bigcypress_plots_deepforest.yaml 0.05
```

Both print per-plot progress including raster dimensions, resolution, dtype, band count and
tile count. **Read the first plot's line** — it confirms which geometry actually ran.

Both write per-plot GeoJSON plus a merged layer. Every point carries a `confidence`
attribute and a `plot` label.

**The campus DeepForest fine-tunes should not be used here.** Fine-tuning specialised them
toward a managed campus, which is the wrong direction for this site. The unmodified release
weights, pretrained on natural forest canopy, are the better-matched starting point.

---

## 8. Choosing a threshold

Detection is run once at a low confidence floor and the score is kept on every point, so
thresholds are explored by **filtering**, not by re-running.

In QGIS: right-click the layer → Filter (or Properties → Source → Query Builder):

```
"confidence" > 0.25
```

To inspect the distribution before choosing:

```bash
python -c "import geopandas as gpd; g=gpd.read_file('./output/yolo_all_plots.geojson'); c=g['confidence']; print('total',len(g)); [print(f'  >{t}: {(c>t).sum():6d}') for t in [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50]]"
```

The two models' confidence scales are **not comparable** — a score of 0.25 means different
things to each, and each needs its own threshold. Provisional values chosen visually are in
`progress_summary.md`.

---

## Appendix: known rough edges

- `nms_iou` is 0.5 in both configs, inherited from the campus pipeline and never swept
  there either. Closed canopy is the case where a lower value should help; this is untested.
- The DeepForest config carries an `imgsz` key for shape consistency with the YOLO config,
  but DeepForest does not use it.