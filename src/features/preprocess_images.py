import utils
import os
import shutil

INPUT_PATH = "../data/interim/"
TRAINING_OUTPUT_PATH = "../data/processed/train"


def setup_output_dirs():
    for class_name in utils.get_indexed_class_names():
        os.makedirs(os.path.join(TRAINING_OUTPUT_PATH, class_name))


def aggregate_training_images():
    for country, regions in utils.LOCATIONS.items():
        for region in regions:
            for class_name in utils.get_indexed_class_names():
                training_path = os.path.join(INPUT_PATH, country, region, "train", class_name)
                for image_name in os.listdir(training_path):
                    image_input_path = os.path.join(training_path, image_name)
                    image_output_path = os.path.join(TRAINING_OUTPUT_PATH, class_name, image_name)
                    shutil.copyfile(image_input_path, image_output_path)


if __name__ == "__main__":
    setup_output_dirs()
    aggregate_training_images()
