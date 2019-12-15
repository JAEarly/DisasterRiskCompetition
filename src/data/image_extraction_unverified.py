"""
Image extraction produces the individual roof top images from the original satellite images.

By using the GeoJson data provided, each roof can be extract from the Tiff satellite images
and saved in smaller, individual files.
"""

import time

import geopandas as gpd
import numpy as np
import os
import rasterio
from PIL import Image
from pandas import DataFrame
from rasterio.mask import mask
from tqdm import tqdm

import utils

INPUT_DIR = "data/raw/stac/"


def extract_images() -> None:
    """
    Run the image extraction for every reason.

    :return: None.
    """
    print("Extracting images")
    for country, regions in utils.LOCATIONS.items():
        for region in regions:
            ImageExtractor.run_for_location("data/raw/stac/", "data/processed/pseudo/all", country, region)


class ImageExtractor():

    def __init__(self, input_dir, output_path, tiff_path, geojson_path):
        self.tiff_path = os.path.join(input_dir, tiff_path)
        self.geojson_path = os.path.join(input_dir, geojson_path)
        self.output_path = output_path
        self.extraction_required = True  # True if extraction should be run

        # Check files exist
        if not os.path.exists(self.tiff_path):
            print("Could not find Tiff, skipping")
            self.extraction_required = False
        if not os.path.exists(self.geojson_path):
            print("Could not find GeoJson, skipping")
            self.extraction_required = False

        if self.extraction_required:
            # Read GeoJson file
            geo_json_dataframe = gpd.read_file(self.geojson_path)

            if 'verified' in geo_json_dataframe.columns:
                geo_json_dataframe = geo_json_dataframe.loc[geo_json_dataframe['verified'] == False]
            else:
                geo_json_dataframe = None

            if geo_json_dataframe is None or len(geo_json_dataframe) == 0:
                print('No unverified images found, skipping')
                self.extraction_required = False
            else:
                # Create and populate new column for projected geometries
                geo_json_dataframe = self.parse_dataframe(geo_json_dataframe)

                # Setup roof geometry dataframe
                self.roof_geometries_dataframe = self.create_roof_geometry_dataframe(
                    geo_json_dataframe
                )

                # Create output dirs if they don't exist
                self.setup_output_dirs()

                # Check if files are already extracted
                num_roofs = len(self.roof_geometries_dataframe.index)
                num_existing = sum(
                    [len(files) for r, d, files in os.walk(self.output_path)]
                )
                if num_existing == num_roofs:
                    self.extraction_required = False
                    print("Already found images")

    def create_roof_geometry_dataframe(self, geo_json_dataframe: DataFrame):
        return geo_json_dataframe[["id", "projected_geometry"]]

    def setup_output_dirs(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_save_path(self, roof):
        return os.path.join(self.output_path, roof.id + ".png")

    def parse_dataframe(self, geo_json_dataframe):
        with rasterio.open(self.tiff_path) as tiff:
            tiff_crs = tiff.crs.data
            geo_json_dataframe["projected_geometry"] = geo_json_dataframe[
                "geometry"
            ].to_crs(tiff_crs)
        return geo_json_dataframe

    def extract_images(self) -> None:
        """
        Run the image extraction. Iterates over each row in the roof dataframe.
        :return: None.
        """
        for _, roof in tqdm(
            self.roof_geometries_dataframe.iterrows(),
            total=len(self.roof_geometries_dataframe.index),
            desc="Extracting images",
            leave=False,
        ):
            roof_image = self.extract_image(roof.id)
            roof_image.save(self.get_save_path(roof))
        time.sleep(1)
        print("")

    def extract_image(self, roof_id: str) -> Image:
        """
        Extract a single image from the Tiff file.
        :param roof_id: Id of the roof to extract.
        :return: The extracted image.
        """
        with rasterio.open(self.tiff_path) as tiff:
            # Get projected geometry for the given roof id
            row = self.roof_geometries_dataframe.loc[
                self.roof_geometries_dataframe["id"] == roof_id
            ]
            projected_geometry = row.projected_geometry.iloc[0]

            # Extract image from tiff file
            roof_image, _ = mask(tiff, [projected_geometry], crop=True, filled=False)

            # Format and return PIL image
            roof_image = np.transpose(roof_image, (1, 2, 0))
            return Image.fromarray(roof_image)

    @staticmethod
    def run_for_location(input_dir: str, output_path: str, country: str, region: str) -> None:
        """
        Run the image extraction for a specific location.
        :param country: Location country.
        :param region: Location region.
        :return: None.
        """
        # Setup paths
        tiff_path = os.path.join(country, region, region + "_ortho-cog.tif")
        geojson_train_path = os.path.join(
            country, region, "train-" + region + ".geojson"
        )

        # Run extraction
        print("Running image extraction for", country, region)
        extractor = ImageExtractor(
            input_dir, output_path, tiff_path, geojson_train_path
        )
        if extractor.extraction_required:
            extractor.extract_images()


if __name__ == "__main__":
    extract_images()
