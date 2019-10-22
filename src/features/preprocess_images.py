import os
import shutil
import time

from tqdm import tqdm

import utils

INPUT_PATH = "data/interim/"
TRAIN_OUTPUT_PATH = "data/processed/train"
TEST_OUTPUT_PATH = "data/processed/test"


def preprocess_images_basic():
    print("Running basic image preprocessing")
    _setup_output_dirs()
    _aggregate_training_images()


def _setup_output_dirs():
    print('Creating output dirs')
    for class_name in utils.get_indexed_class_names():
        path = os.path.join(TRAIN_OUTPUT_PATH, class_name)
        if not os.path.exists(path):
            os.makedirs(path)
    if not os.path.exists(TEST_OUTPUT_PATH):
        os.makedirs(TEST_OUTPUT_PATH)


def _aggregate_training_images():
    print('Aggregating training images')
    for country, regions in utils.LOCATIONS.items():
        for region in regions:
            # Copy training images
            for class_name in tqdm(utils.get_indexed_class_names(), desc=country + " " + region + " train"):
                training_path = os.path.join(INPUT_PATH, country, region, "train", class_name)
                for image_name in os.listdir(training_path):
                    image_input_path = os.path.join(training_path, image_name)
                    image_output_path = os.path.join(TRAIN_OUTPUT_PATH, class_name, image_name)
                    shutil.copyfile(image_input_path, image_output_path)
            time.sleep(1)
            # Copy test images
            test_path = os.path.join(INPUT_PATH, country, region, "test")
            if os.path.exists(test_path):
                for image_name in tqdm(os.listdir(test_path), desc=country + " " + region + " test"):
                    image_input_path = os.path.join(test_path, image_name)
                    image_output_path = os.path.join(TEST_OUTPUT_PATH, image_name)
                    shutil.copyfile(image_input_path, image_output_path)
            time.sleep(1)
