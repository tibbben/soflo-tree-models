# soflo-tree-models

A playground for image segmentation to detect the trees of South Florida.

- ./bicy_share.qgs
  is a QGIS file that will open if you place the download folder from the [shared BOX folder](https://miami.box.com/s/qd4j3x2pf0v2gn9knjz9r9ffnsyku4wx) at the top level of this repo. You can visualize the cropped images, the approximate survey locations, and the historical survey data.

- ./jupyter/crop_orthoimages.ipynb
  a code generator that creates a set of shell scripts to crop the original orthophotos to approximate site locations.

- .jupyter/csv_to_shapefile.ipynb
  an example of how to read csv into a geopandas dataframe