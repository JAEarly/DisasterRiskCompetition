from data.make_dataset import download_dataset, extract_dataset
from data.image_extraction import extract_images
from data.preprocess_images import preprocess_images_basic, validate_dataset

num_stages = 5
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
print("Done")
