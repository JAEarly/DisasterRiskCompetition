"""Setup script."""

from data.make_dataset import download_dataset, extract_dataset
from data.image_extraction import extract_images
from data.preprocess_images import preprocess_images_basic, validate_dataset
from model_manager import ModelManager
from data.augment_dataset import augment_dataset

num_stages = 7
print("Starting setup")
print("\n(1/" + str(num_stages) + " - Download Dataset)")
download_dataset()
print("\n(2/" + str(num_stages) + " - Extract Dataset)")
extract_dataset()
print("\n(3/" + str(num_stages) + " - Extract Images)")
extract_images()
print("\n(4/" + str(num_stages) + " - Preprocess Images)")
preprocess_images_basic()
print("\n(5/" + str(num_stages) + " - Validate Dataset)")
validate_dataset()
print("\n(6/" + str(num_stages) + " - Augmenting Dataset)")
augment_dataset()
print("\n(7/" + str(num_stages) + " - Downloading Models")
ModelManager().download_all()
print("Done")
