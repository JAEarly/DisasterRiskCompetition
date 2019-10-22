from data.make_dataset import download_dataset, extract_dataset
from data.image_extraction import extract_images
from features.preprocess_images import preprocess_images_basic

print("Starting setup")
download_dataset()
extract_dataset()
extract_images()
preprocess_images_basic()
print("Done")
