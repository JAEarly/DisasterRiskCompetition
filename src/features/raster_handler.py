import os

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.mask import mask
from tqdm import tqdm

import utils

INPUT_DIR = "../data/raw/stac/"
OUTPUT_DIR = "../data/interim/"


class RasterHandler:

    def __init__(self, tiff_rel_path, geojson_rel_path):
        self.tiff_path = os.path.join(INPUT_DIR, tiff_rel_path)
        self.geojson_path = os.path.join(INPUT_DIR, geojson_rel_path)

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
            df_roof_geometries[['id', 'roof_material', 'projected_geometry']]
        )

        # Ensure output directories exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            for indexed_class_name in utils.get_indexed_class_names():
                os.makedirs(os.path.join(OUTPUT_DIR, indexed_class_name))

    def create_dataset(self):
        for _, roof in tqdm(self.roof_geometries.iterrows(),
                            total=len(self.roof_geometries.index),
                            desc="Extracting images"):
            roof_image = self.extract_image(roof.id)
            roof_image.save(os.path.join(OUTPUT_DIR,
                                         utils.get_indexed_class_name(roof.roof_material),
                                         roof.id + ".png"))

    def extract_image(self, roof_id):
        with rasterio.open(self.tiff_path) as tiff:
            # Get projected geometry for the given roof id
            df = self.roof_geometries.loc[self.roof_geometries['id'] == roof_id]
            projected_geometry = df.projected_geometry.iloc[0]

            # Extract image from tiff file
            roof_image, _ = mask(
                tiff, [projected_geometry], crop=True, filled=False
            )

            # Format and return PIL image
            roof_image = np.transpose(roof_image, (1, 2, 0))
            return Image.fromarray(roof_image)


if __name__ == "__main__":
    _tiff_rel_path = "colombia/borde_rural/borde_rural_ortho-cog.tif"
    _geojson_rel_path = "colombia/borde_rural/train-borde_rural.geojson"
    rh = RasterHandler(_tiff_rel_path, _geojson_rel_path)
    rh.create_dataset()







