import os
import time
from abc import ABC, abstractmethod

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.mask import mask
from tqdm import tqdm

import utils

INPUT_DIR = "../data/raw/stac/"
OUTPUT_DIR = "../data/interim/"


class ImageExtractor(ABC):

    def __init__(self, tiff_path, geojson_path, output_path):
        self.tiff_path = os.path.join(INPUT_DIR, tiff_path)
        self.geojson_path = os.path.join(INPUT_DIR, geojson_path)
        self.output_path = os.path.join(OUTPUT_DIR, output_path)
        self.files_valid = True

        print('Tiff:', self.tiff_path)
        if not os.path.exists(self.tiff_path):
            print('Could not find Tiff, skipping')
            self.files_valid = False
        print('GeoJson:', self.geojson_path)
        if not os.path.exists(self.geojson_path):
            print('Could not find GeoJson, skipping')
            self.files_valid = False
        print('Output:', self.output_path)

        # Read GeoJson file
        if self.files_valid:
            df_roof_geometries = gpd.read_file(self.geojson_path)

            # Create new column in for projected geometries
            with rasterio.open(self.tiff_path) as tiff:
                tiff_crs = tiff.crs.data
                df_roof_geometries['projected_geometry'] = (
                    df_roof_geometries['geometry'].to_crs(tiff_crs)
                )

            self.roof_geometries = self.create_roof_geometry_dataframe(df_roof_geometries)
            self.setup_output_dirs()

    @abstractmethod
    def create_roof_geometry_dataframe(self, df_roof_geometries):
        pass

    @abstractmethod
    def setup_output_dirs(self):
        pass

    @abstractmethod
    def get_save_path(self, roof):
        pass

    def create_dataset(self):
        for _, roof in tqdm(self.roof_geometries.iterrows(),
                            total=len(self.roof_geometries.index),
                            desc="Extracting images"):
            roof_image = self.extract_image(roof.id)
            roof_image.save(self.get_save_path(roof))
        time.sleep(1)
        print('')

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

    @staticmethod
    def run_for_location(country, region):
        tiff_path = os.path.join(country, region, region + "_ortho-cog.tif")
        geojson_train_path = os.path.join(country, region, "train-" + region + ".geojson")
        geojson_test_path = os.path.join(country, region, "test-" + region + ".geojson")
        output_train_path = os.path.join(country, region, "train")
        output_test_path = os.path.join(country, region, "test")

        print('Running image extraction for', country, region, "train")
        extractor = TrainImageExtractor(tiff_path, geojson_train_path, output_train_path)
        if extractor.files_valid:
            extractor.create_dataset()

        print('Running image extraction for', country, region, "test")
        extractor = TestImageExtractor(tiff_path, geojson_test_path, output_test_path)
        if extractor.files_valid:
            extractor.create_dataset()


class TrainImageExtractor(ImageExtractor):

    def create_roof_geometry_dataframe(self, df_roof_geometries):
        return (
            df_roof_geometries[['id', 'roof_material', 'projected_geometry']]
        )

    def setup_output_dirs(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            for indexed_class_name in utils.get_indexed_class_names():
                os.makedirs(os.path.join(self.output_path, indexed_class_name))

    def get_save_path(self, roof):
        return os.path.join(self.output_path,
                     utils.get_indexed_class_name(roof.roof_material),
                     roof.id + ".png")


class TestImageExtractor(ImageExtractor):

    def create_roof_geometry_dataframe(self, df_roof_geometries):
        return (
            df_roof_geometries[['id', 'projected_geometry']]
        )

    def setup_output_dirs(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_save_path(self, roof):
        return os.path.join(self.output_path,
                            roof.id + ".png")


if __name__ == "__main__":
    for country, regions in utils.LOCATIONS.items():
        for region in regions:
            ImageExtractor.run_for_location(country, region)
