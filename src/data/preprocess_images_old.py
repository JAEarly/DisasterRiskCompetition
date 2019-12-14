"""Preprocess extracted roof images."""

import os
import random
import shutil
import time

from texttable import Texttable
from tqdm import tqdm

import utils

INPUT_PATH = "data/interim_old/"
TRAIN_OUTPUT_PATH = "data/processed_old/train"
VALIDATION_OUTPUT_PATH = "data/processed_old/validation"
TEST_OUTPUT_PATH = "data/processed_old/test"
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
    for class_name in utils.get_indexed_class_names_old():
        utils.create_dirs_if_not_found(os.path.join(TRAIN_OUTPUT_PATH, class_name))
        utils.create_dirs_if_not_found(os.path.join(VALIDATION_OUTPUT_PATH, class_name))
        utils.create_dirs_if_not_found(os.path.join(TEST_OUTPUT_PATH, class_name))


def _aggregate_images() -> None:
    """
    Aggregate the images in per location folders to single class folders.

    Also creates the train/validation/test split.
    :return: None
    """
    print("Aggregating training images")
    for grid_idx in range(12):
        grid_name = "grid_" + str(grid_idx)
        print('Aggregating for', grid_name)
        for class_name in utils.get_indexed_class_names_old():
            interim_input_path = os.path.join(
                INPUT_PATH, grid_name, class_name
            )
            files = os.listdir(interim_input_path)
            random.shuffle(files)
            for idx, image_name in enumerate(files):
                # Work out if image should be given to train, validation or test based on splits
                i = idx / len(files)
                if i < (1 - VALIDATION_SPLIT - TEST_SPLIT):
                    base_path = TRAIN_OUTPUT_PATH
                elif i < (1 - TEST_SPLIT):
                    base_path = VALIDATION_OUTPUT_PATH
                else:
                    base_path = TEST_OUTPUT_PATH

                # Copy image to relevant folder
                image_input_path = os.path.join(interim_input_path, image_name)
                image_output_path = os.path.join(base_path, class_name, grid_name + "_" + image_name)
                shutil.copyfile(image_input_path, image_output_path)
        time.sleep(0.1)


def validate_dataset() -> None:
    """
    Validate the image preprocessing was correct and see the data splits.
    :return: None.
    """
    print("   --- Dataset Validation ---")
    num_train_images = utils.count_files_recursive(TRAIN_OUTPUT_PATH)
    num_validation_images = utils.count_files_recursive(VALIDATION_OUTPUT_PATH)
    num_test_images = utils.count_files_recursive(TEST_OUTPUT_PATH)
    num_total_images = utils.count_files_recursive(INPUT_PATH)

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

    table = Texttable()
    rows = [["Category", *utils.get_indexed_class_names_old(), "Total"]]
    table.set_cols_align(["c", "c", "c", "c", "c"])
    table.set_cols_width([18, 18, 18, 18, 18])
    for category, path in [
        ("Training", TRAIN_OUTPUT_PATH),
        ("Validation", VALIDATION_OUTPUT_PATH),
        ("Test", TEST_OUTPUT_PATH),
    ]:
        row = [category]
        total_count = utils.count_files_recursive(path)
        for class_name in utils.get_indexed_class_names_old():
            class_dir = os.path.join(path, class_name)
            class_count = utils.count_files_recursive(class_dir)
            class_percentage = class_count / total_count * 100
            row.append("{:d} ({:.2f}%)".format(class_count, class_percentage))
        row.append(total_count)
        rows.append(row)
    table.add_rows(rows)
    print(table.draw())


if __name__ == "__main__":
    preprocess_images_basic()
    validate_dataset()
