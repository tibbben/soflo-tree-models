# This code explores the outputs of the detection and segmentation models (out-of-the-box models)
# that were originally run on the data 

import geopandas as gpd

det = gpd.read_file("./download/um_gables_tree_detection.geojson")
seg = gpd.read_file("./download/um_gables_tree_segmentation.geojson")

print("=== DETECTION ===")
print(f"Total features: {len(det)}")
print(f"Geometry type: {det.geometry.geom_type.unique()}")
print(f"Columns: {det.columns.tolist()}")
print(det.head(3))

print("\n=== SEGMENTATION ===")
print(f"Total features: {len(seg)}")
print(f"Geometry type: {seg.geometry.geom_type.unique()}")
print(f"Columns: {seg.columns.tolist()}")
print(seg.head(3))