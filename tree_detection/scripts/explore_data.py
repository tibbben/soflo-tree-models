# This code loads um_gables_trees.geojson (the labelled data)
# and prints summary stats: total trees, column names, top 20 species, pine/cypress counts, canopy area stats

import geopandas as gpd

gdf = gpd.read_file("./download/um_gables_trees.geojson")

print(f"Total trees: {len(gdf)}")
print(f"\nColumns: {gdf.columns.tolist()}")
print(f"\nTop 20 common names:\n{gdf['common_name'].value_counts().head(20)}")
print(f"\nCanopy area null count: {gdf['canopy_area_sqm'].isna().sum()}")

# Check for pine and cypress
print("\nPine trees:")
print(gdf[gdf['common_name'].str.contains('Pine', case=False, na=False)]['common_name'].value_counts())

print("\nCypress trees:")
print(gdf[gdf['common_name'].str.contains('Cypress', case=False, na=False)]['common_name'].value_counts())

print("\nPalm trees:")
print(gdf[gdf['tax_family'].str.contains('Arecaceae', case=False, na=False)]['common_name'].value_counts())

# Check canopy area stats for trees that have it
print(f"\nCanopy area stats (where available):")
print(gdf['canopy_area_sqm'].describe())