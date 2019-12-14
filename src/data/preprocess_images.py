"""Preprocess extracted roof images."""

import os
import random
import shutil
import time

from texttable import Texttable
from tqdm import tqdm

import utils

INPUT_PATH = "data/interim/"
TRAIN_OUTPUT_PATH = "data/processed/train"
VALIDATION_OUTPUT_PATH = "data/processed/validation"
TEST_OUTPUT_PATH = "data/processed/test"
COMPETITION_OUTPUT_PATH = "data/processed/competition/all"
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1


def preprocess_images_basic() -> None:
    """
    Run the basic preprocessing.
    :return: None.
    """
    print("Running basic image preprocessing")
    _setup_output_dirs()
    _aggregate_images()


def _setup_output_dirs() -> None:
    """
    Create output dirs.
    :return: None.
    """
    print("Creating output dirs")
    for class_name in utils.get_indexed_class_names():
        utils.create_dirs_if_not_found(os.path.join(TRAIN_OUTPUT_PATH, class_name))
        utils.create_dirs_if_not_found(os.path.join(VALIDATION_OUTPUT_PATH, class_name))
        utils.create_dirs_if_not_found(os.path.join(TEST_OUTPUT_PATH, class_name))
    utils.create_dirs_if_not_found(COMPETITION_OUTPUT_PATH)


def _aggregate_images() -> None:
    """
    Aggregate the images in per location folders to single class folders.

    Also creates the train/validation/test split.
    :return: None
    """
    print("Aggregating training images")
    for country, regions in utils.LOCATIONS.items():
        for region in regions:

            # Copy training images
            for class_name in utils.get_indexed_class_names():
                training_path = os.path.join(
                    INPUT_PATH, country, region, "train", class_name
                )
                if not os.path.exists(training_path):
                    continue
                files = os.listdir(training_path)
                random.shuffle(files)
                for idx, image_name in enumerate(
                    tqdm(
                        files,
                        desc=country + " " + region + " train " + class_name,
                        leave=False,
                    )
                ):
                    # Work out if image should be given to train, validation or test based on splits
                    i = idx / len(files)
                    if i < (1 - VALIDATION_SPLIT - TEST_SPLIT):
                        base_path = TRAIN_OUTPUT_PATH
                    elif i < (1 - TEST_SPLIT):
                        base_path = VALIDATION_OUTPUT_PATH
                    else:
                        base_path = TEST_OUTPUT_PATH

                    # Copy image to relevant folder
                    image_input_path = os.path.join(training_path, image_name)
                    image_output_path = os.path.join(base_path, class_name, image_name)
                    shutil.copyfile(image_input_path, image_output_path)
            time.sleep(0.1)

            # Copy competition images
            competition_path = os.path.join(INPUT_PATH, country, region, "test")
            if os.path.exists(competition_path):
                for image_name in tqdm(
                    os.listdir(competition_path),
                    desc=country + " " + region + " test",
                    leave=False,
                ):
                    image_input_path = os.path.join(competition_path, image_name)
                    image_output_path = os.path.join(
                        COMPETITION_OUTPUT_PATH, image_name
                    )
                    shutil.copyfile(image_input_path, image_output_path)
            time.sleep(1)


def validate_dataset() -> None:
    """
    Validate the image preprocessing was correct and see the data splits.
    :return: None.
    """
    print("   --- Dataset Validation ---")
    num_train_images = utils.count_files_recursive(TRAIN_OUTPUT_PATH)
    num_validation_images = utils.count_files_recursive(VALIDATION_OUTPUT_PATH)
    num_test_images = utils.count_files_recursive(TEST_OUTPUT_PATH)
    num_total_images = utils.count_files_recursive(INPUT_PATH, contains="train")

    num_competition_images = utils.count_files_recursive(COMPETITION_OUTPUT_PATH)

    print(
        "      Train:",
        num_train_images,
        " Split:",
        "{:.3f}".format(num_train_images / num_total_images),
    )
    print(
        " Validation: ",
        num_validation_images,
        " Split:",
        "{:.3f}".format(num_validation_images / num_total_images),
    )
    print(
        "       Test: ",
        num_test_images,
        " Split:",
        "{:.3f}".format(num_test_images / num_total_images),
    )
    print("      Total:", num_total_images)

    assert (
        num_train_images + num_validation_images + num_test_images == num_total_images
    )
    print("Competition:", num_competition_images)

    table = Texttable()
    rows = [["Category", *utils.get_indexed_class_names(), "Total"]]
    table.set_cols_align(["c", "c", "c", "c", "c", "c", "c"])
    table.set_cols_width([18, 18, 18, 18, 18, 18, 18])
    for category, path in [
        ("Training", TRAIN_OUTPUT_PATH),
        ("Validation", VALIDATION_OUTPUT_PATH),
        ("Test", TEST_OUTPUT_PATH),
    ]:
        row = [category]
        total_count = utils.count_files_recursive(path)
        for class_name in utils.get_indexed_class_names():
            class_dir = os.path.join(path, class_name)
            class_count = utils.count_files_recursive(class_dir)
            class_percentage = class_count / total_count * 100
            row.append("{:d} ({:.2f}%)".format(class_count, class_percentage))
        row.append(total_count)
        rows.append(row)
    table.add_rows(rows)
    print(table.draw())
