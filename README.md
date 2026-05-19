# soflo-tree-models

A playground for image segmentation to detect the trees of South Florida.

## maps

- ./bicy_share.qgs
  is a QGIS file that will open if you place the download folder from the [shared Big Cypress folder](https://miami.box.com/s/qd4j3x2pf0v2gn9knjz9r9ffnsyku4wx) at the top level of this repo. You can visualize the cropped images, the approximate survey locations, and the historical survey data.

- ./gables_share.qgs
  is a QGIS file that will open if you place the download folder from the [shared Gables Campus folder](https://miami.box.com/s/mq6k0vj8f89h4w91u7pqdocetjpoma2f) at the top level of this repo. You can visualize the drone survey, the tree inventory, and the deep learning models from ArcGIS Pro.

## jupyter

- ./jupyter/crop_orthoimages.ipynb
  a code generator that creates a set of shell scripts to crop the original Big Cypress orthophotos to approximate site locations.

- .jupyter/csv_to_shapefile.ipynb
  an example of how to read csv into a geopandas dataframe (from Big Cypress historical survey)

### Resources

- [The root shared Box folder](https://miami.box.com/s/qtijqu7e5g36wircuzts951mk8oqwyj3)
   - [Running Notes](https://miami.box.com/s/86s4e45lwxvta9lh5ow5mlze72ngh5v9)
   - [The shared Gables Campus folder](https://miami.box.com/s/mq6k0vj8f89h4w91u7pqdocetjpoma2f)
   - [The shared Big Cypress Folder](https://miami.box.com/s/qd4j3x2pf0v2gn9knjz9r9ffnsyku4wx)
   - [The GDSC metadata spreadsheet](https://miami.box.com/s/cpe136whxprafac9ssvkig74ju4o2x7m)