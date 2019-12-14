import csv

import os
import requests
from tqdm import tqdm

from utils import create_dirs_if_not_found

LINKS_PATH = "./data/raw/old_dataset_links.csv"


def get_download_links():
    links = []
    with open(LINKS_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        # Skip header
        next(csv_reader, None)
        for row in csv_reader:
            links.append(row)
    return links


def download_files(download_links):
    print("Downloading all files")
    create_dirs_if_not_found("./data/raw/old")
    grid_id = 1
    for geojson_link, geotiff_link in download_links:
        download_file(geojson_link, "./data/raw/old/grid_" + str(grid_id) + ".json")
        download_file(geotiff_link, "./data/raw/old/grid_" + str(grid_id) + ".tiff")
        grid_id += 1
    print("Done")


def download_file(link, path) -> None:
    # Only download dataset if not found
    if not os.path.exists(path):
        print("Downloading", path)
        request = requests.get(link, stream=True)

        # Total size in bytes.
        total_size = int(request.headers.get("content-length", 0))
        block_size = 1024

        # Download with tqdm progress bar
        p_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(path, "wb") as output_file:
            for data in request.iter_content(block_size):
                p_bar.update(len(data))
                output_file.write(data)
        p_bar.close()
    else:
        print("Already found", path, "- skipping download")


if __name__ == "__main__":
    download_files(get_download_links())
