from rasterio.mask import mask

import rasterio
import os
import json

import geopandas as gpd
import numpy as np

from PIL import Image


ROOT_DIR = "../data/stac/"


class RasterHandler:

    def __init__(self, tiff_rel_path, geojson_rel_path):
        self.tiff_path = os.path.join(ROOT_DIR, tiff_rel_path)
        self.geojson_path = os.path.join(ROOT_DIR, geojson_rel_path)

        # Read GeoJson file
        df_roof_geometries = gpd.read_file(self.geojson_path)

        # Create new column in for projected geometries
        with rasterio.open(self.tiff_path) as tiff:
            tiff_crs = tiff.crs.data
            df_roof_geometries['projected_geometry'] = (
                df_roof_geometries['geometry'].to_crs(tiff_crs)
            )

        # Reduce df to id and projected geometries only
        self.roof_geometries = (
            df_roof_geometries[['id', 'projected_geometry']]
        )

    def extract_image(self, roof_id):
        with rasterio.open(self.tiff_path) as tiff:
            print(roof_id)
            df = self.roof_geometries.loc[self.roof_geometries['id'] == roof_id]
            print(df)
            projected_geometry = df.projected_geometry.iloc[0]
            print(projected_geometry)

            roof_image, _ = mask(
                tiff, [projected_geometry], crop=True, filled=False
            )
            roof_image = np.transpose(roof_image, (1, 2, 0))

            Image.fromarray(roof_image).show()

    def extract_images(self):
        for i in range(5):
            roof_id = self.roof_geometries.iloc[i].id
            self.extract_image(roof_id)


if __name__ == "__main__":
    _tiff_rel_path = "colombia/borde_rural/borde_rural_ortho-cog.tif"
    _geojson_rel_path = "colombia/borde_rural/train-borde_rural.geojson"
    rh = RasterHandler(_tiff_rel_path, _geojson_rel_path)
    rh.extract_images()







