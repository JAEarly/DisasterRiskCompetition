"""
Image extraction produces the individual roof top images from the original satellite images.

By using the GeoJson data provided, each roof can be extract from the Tiff satellite images
and saved in smaller, individual files.
"""

import os
import time
from abc import ABC, abstractmethod

import geopandas as gpd
from pandas import DataFrame
import numpy as np
import rasterio
from PIL import Image
from rasterio.mask import mask
from tqdm import tqdm

import utils


INPUT_DIR = "data/raw/stac/"
OUTPUT_DIR = "data/interim/"


def extract_images() -> None:
    """
    Run the image extraction for every reason.

    :return: None.
    """
    print("Extracting images")
    for country, regions in utils.LOCATIONS.items():
        for region in regions:
            ImageExtractor.run_for_location(country, region)


class ImageExtractor(ABC):
    """Base class for image extraction."""

    def __init__(self, tiff_path, geojson_path, output_path, verified_only=True):
        self.tiff_path = os.path.join(INPUT_DIR, tiff_path)
        self.geojson_path = os.path.join(INPUT_DIR, geojson_path)
        self.output_path = os.path.join(OUTPUT_DIR, output_path)
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
            if verified_only and 'verified' in geo_json_dataframe.columns:
                geo_json_dataframe = geo_json_dataframe.loc[geo_json_dataframe['verified'] == True]

            if len(geo_json_dataframe) == 0:
                print('No verified images found, skipping')
                self.extraction_required = False
            else:
                # Create and populate new column for projected geometries
                with rasterio.open(self.tiff_path) as tiff:
                    tiff_crs = tiff.crs.data
                    geo_json_dataframe["projected_geometry"] = geo_json_dataframe[
                        "geometry"
                    ].to_crs(tiff_crs)

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

    @abstractmethod
    def create_roof_geometry_dataframe(
        self, geo_json_dataframe: DataFrame
    ) -> DataFrame:
        """
        Create the roof geometry dataframe from a geo json dataframe.
        :param geo_json_dataframe: Dataframe containing information from the geo json file.
        :return: The reduced roof geometry dataframe.
        """

    @abstractmethod
    def setup_output_dirs(self) -> None:
        """
        Create the necessary output directories.
        :return: None.
        """

    @abstractmethod
    def get_save_path(self, roof) -> str:
        """
        Get the save path for a roof.
        :param roof: Roof entry from dataframe table.
        :return: Save path as string.
        """

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
    def run_for_location(country: str, region: str) -> None:
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
        geojson_test_path = os.path.join(country, region, "test-" + region + ".geojson")
        output_train_path = os.path.join(country, region, "train")
        output_test_path = os.path.join(country, region, "test")

        # Run training (labelled data) extraction
        print("Running image extraction for", country, region, "train")
        extractor = LabelledImageExtractor(
            tiff_path, geojson_train_path, output_train_path
        )
        if extractor.extraction_required:
            extractor.extract_images()

        # Run test (unlabelled data) extraction
        print("Running image extraction for", country, region, "test")
        extractor = UnlabelledImageExtractor(
            tiff_path, geojson_test_path, output_test_path
        )
        if extractor.extraction_required:
            extractor.extract_images()


class LabelledImageExtractor(ImageExtractor):
    """
    Extractor for labelled images.

    Roof dataframe contains a label for the roof material, and images are saved into class dirs.
    """

    def create_roof_geometry_dataframe(self, geo_json_dataframe: DataFrame):
        return geo_json_dataframe[["id", "roof_material", "projected_geometry"]]

    def setup_output_dirs(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            # Make dirs based on classes
            for indexed_class_name in utils.get_indexed_class_names():
                os.makedirs(os.path.join(self.output_path, indexed_class_name))

    def get_save_path(self, roof):
        # Save into class dir
        return os.path.join(
            self.output_path,
            utils.get_indexed_class_name(roof.roof_material),
            roof.id + ".png",
        )


class UnlabelledImageExtractor(ImageExtractor):
    """
    Extractor for unlabelled images.

    No class label in roof dataframe and images all saved in single dir.
    """

    def create_roof_geometry_dataframe(self, geo_json_dataframe: DataFrame):
        return geo_json_dataframe[["id", "projected_geometry"]]

    def setup_output_dirs(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_save_path(self, roof):
        return os.path.join(self.output_path, roof.id + ".png")
