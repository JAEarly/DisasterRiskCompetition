from tqdm import tqdm
import requests
import tarfile
import os


DATASET_URL = "https://drivendata-public-assets-eu.s3.eu-central-1.amazonaws.com/stac.tar"
DATASET_PATH = "../data/stac.tar"
EXTRACT_PATH = "../data/"


def download_dataset():
    # Only download dataset if not found
    if not os.path.exists(DATASET_PATH):
        print("Downloading dataset")
        r = requests.get(DATASET_URL, stream=True)

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024

        # Download with tqdm progress bar
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(DATASET_PATH, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
    else:
        print("Dataset file found, skipping download")


def extract_dataset():
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
