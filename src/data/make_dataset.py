"""Dataset setup - download and extract from tar file."""

import os
import tarfile

import requests
from tqdm import tqdm

DATASET_URL = (
    "https://drivendata-public-assets-eu.s3.eu-central-1.amazonaws.com/stac.tar"
)
DATASET_PATH = "data/raw/stac.tar"
EXTRACT_PATH = "data/raw/"


def download_dataset() -> None:
    """
    Download the dataset from the internet. Takes a while.
    :return: None.
    """
    # Only download dataset if not found
    if not os.path.exists(DATASET_PATH):
        print("Downloading dataset")
        request = requests.get(DATASET_URL, stream=True)

        # Total size in bytes.
        total_size = int(request.headers.get("content-length", 0))
        block_size = 1024

        # Download with tqdm progress bar
        p_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(DATASET_PATH, "wb") as output_file:
            for data in request.iter_content(block_size):
                p_bar.update(len(data))
                output_file.write(data)
        p_bar.close()
    else:
        print("Dataset file found, skipping download")


def extract_dataset() -> None:
    """
    Extract dataset from tar file.
    :return: None.
    """
    # Only extract dataset if not already extracted
    if not os.path.exists(os.path.join(EXTRACT_PATH, "stac")):
        print("Extracting dataset")
        if not os.path.exists(EXTRACT_PATH):
            os.makedirs(EXTRACT_PATH)
        tar = tarfile.open(DATASET_PATH)
        tar.extractall(path=EXTRACT_PATH)
        tar.close()
    else:
        print("Extract dataset found, skipping extraction")
