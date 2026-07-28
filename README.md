\# soflo-tree-models



A playground for image segmentation to detect the trees of South Florida.



\## maps



\* ./bicy\_share.qgs is a QGIS file that will open if you place the download folder from the \[shared Big Cypress folder](https://miami.box.com/s/qd4j3x2pf0v2gn9knjz9r9ffnsyku4wx) at the top level of this repo. You can visualize the cropped images, the approximate survey locations, and the historical survey data.

\* ./gables\_share.qgs is a QGIS file that will open if you place the download folder from the \[shared Gables Campus folder](https://miami.box.com/s/mq6k0vj8f89h4w91u7pqdocetjpoma2f) at the top level of this repo. You can visualize the drone survey, the tree inventory, and the deep learning models from ArcGIS Pro.



\## jupyter



\* ./jupyter/crop\_orthoimages.ipynb a code generator that creates a set of shell scripts to crop the original Big Cypress orthophotos to approximate site locations.

\* ./jupyter/csv\_to\_shapefile.ipynb an example of how to read csv into a geopandas dataframe (from Big Cypress historical survey)

\* ./jupyter/combine\_shapefiles.ipynb combines a set of shapefiles into a single geopackage.

\* ./jupyter/convert\_census\_to\_geojson.ipynb converts the Big Cypress plot census CSV into a georeferenced GeoJSON point layer, anchoring each tree's local plot coordinates to its plot centre in UTM.



\## tree\_detection



Deep learning pipelines that detect individual trees and export one point per tree for GIS use. Two sites, each with its own subfolder, README, and write-up.



\* \[./tree\_detection/gables\_campus](tree\_detection/gables\_campus) — A YOLO detection pipeline trained on the Coral Gables campus drone survey against a 10,659-point tree inventory. The champion model reaches F1 0.660 on the evaluation region, compared to 0.401 for the prior ArcGIS Pro baseline on the same ground — roughly a 65% improvement in F1 and 2.45× as many trees found. Includes a reproducible config-driven pipeline, benchmarking, and a record of what was tried.

\* \[./tree\_detection/big\_cypress](tree\_detection/big\_cypress) — The campus champion and a forest-pretrained DeepForest model applied to the Big Cypress plot clips. No usable ground truth exists for this site yet, so results are visual only.



\## Resources



\* \[The root shared Box folder](https://miami.box.com/s/qtijqu7e5g36wircuzts951mk8oqwyj3)

&#x20; \* \[Running Notes](https://miami.box.com/s/86s4e45lwxvta9lh5ow5mlze72ngh5v9)

&#x20; \* \[The shared Gables Campus folder](https://miami.box.com/s/mq6k0vj8f89h4w91u7pqdocetjpoma2f)

&#x20; \* \[The shared Big Cypress Folder](https://miami.box.com/s/qd4j3x2pf0v2gn9knjz9r9ffnsyku4wx)

&#x20; \* \[The GDSC metadata spreadsheet](https://miami.box.com/s/cpe136whxprafac9ssvkig74ju4o2x7m)

