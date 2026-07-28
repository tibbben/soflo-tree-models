"""
benchmark_gtregion.py — Score detections against the ground-truth tree points.

A detection counts as a TRUE POSITIVE if it falls within MATCH_DIST meters of a
ground-truth tree. Matching is one-to-one: highest-confidence detections are matched
first, each ground-truth tree can be claimed once, and the nearest eligible tree
wins. Reports precision / recall / F1.

Scores against the EVALUATION-REGION ground truth (the tree points clipped to the
hand-drawn polygon covering the labelled area). Detections must come from a matching
evaluation-region run — scoring a full-survey detection file against this ground truth
would count every correctly-detected tree outside the polygon as a false positive.

Works whether the detection file is POINTS or POLYGONS (polygons reduced to centroids),
so alternative detectors can be scored on the same footing.

Run from the project root:
    python scripts/benchmark_gtregion.py <detections.geojson> [match_dist_m] [min_conf]

    detections : e.g. ./output/tree_detections_gtregion.geojson
    match_dist : match radius in meters (default 5.0, which is what every recorded
                 result used)
    min_conf   : OPTIONAL confidence floor (default 0.0 = keep all). Drops detections
                 below this confidence before scoring, so a single low-confidence
                 detection run supports an entire threshold sweep without re-running
                 inference. Confidence does not change the model — it selects an
                 operating point, so models must be compared each at its own peak-F1
                 confidence.
"""

import sys
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

# Args (parameterized by sys.argv, not argparse)
if len(sys.argv) < 2:
    sys.exit("Usage: python scripts/benchmark_gtregion.py <detections.geojson> "
             "[match_dist_m] [min_conf]")
det_path = sys.argv[1]
match_dist = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
min_conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # 0 = keep everything

# ground truth clipped to the evaluation polygon
gt_path = "./download/um_gables_trees_gtregion.geojson"

# Force both layers into the survey's metric CRS so distances are in METERS.
METRIC_CRS = "EPSG:32617"
det = gpd.read_file(det_path).to_crs(METRIC_CRS)
gt = gpd.read_file(gt_path).to_crs(METRIC_CRS)

# Optional confidence floor — sweep operating points without re-running detection.
if min_conf > 0.0:
    if "confidence" not in det.columns:
        sys.exit("min_conf was given but the detection file has no 'confidence' column.")
    before = len(det)
    det = det[det["confidence"] >= min_conf].reset_index(drop=True)
    print(f"Confidence floor {min_conf}: kept {len(det)} of {before} detections")

# Match highest-confidence detections first, if a confidence column exists.
if "confidence" in det.columns:
    det = det.sort_values("confidence", ascending=False).reset_index(drop=True)

# Centroids work for POINT or POLYGON inputs. CRS is metric, so this is exact.
gt_xy = np.array([(g.centroid.x, g.centroid.y) for g in gt.geometry])
det_xy = np.array([(g.centroid.x, g.centroid.y) for g in det.geometry])

# KD-tree on ground-truth points for fast radius lookups.
kdt = cKDTree(gt_xy)

matched_gt = set()   # ground-truth indices already claimed
tp = 0

for dxy in det_xy:
    near = kdt.query_ball_point(dxy, r=match_dist)
    near = [i for i in near if i not in matched_gt]
    if near:
        best = min(near, key=lambda i: (gt_xy[i][0] - dxy[0])**2 + (gt_xy[i][1] - dxy[1])**2)
        matched_gt.add(best)
        tp += 1

n_det = len(det_xy)
n_gt = len(gt_xy)
fp = n_det - tp
fn = n_gt - tp
precision = tp / n_det if n_det else 0.0
recall = tp / n_gt if n_gt else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"Detections: {n_det} | Ground truth: {n_gt} | Match radius: {match_dist} m")
print(f"True positives:  {tp}")
print(f"False positives: {fp}  (detections with no tree nearby)")
print(f"False negatives: {fn}  (real trees the model missed)")
print(f"Precision: {precision:.3f}   (of detections, how many are real)")
print(f"Recall:    {recall:.3f}   (of real trees, how many were found)")
print(f"F1:        {f1:.3f}")