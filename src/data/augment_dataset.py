import numpy as np
import os
import random
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

import utils
from utils import create_dirs_if_not_found

base_dir = "./data/processed/train"
output_dir = "./data/augmented/train"


def augment_dataset():
    print('Running dataset augmentation for', base_dir)
    clz_counts = utils.class_distribution(base_dir)
    print('Original counts:', clz_counts)
    num_outputs = max(clz_counts)
    for clz in utils.get_indexed_class_names():
        augment_all_in_dir(clz, num_outputs)
    clz_counts = utils.class_distribution(output_dir)
    print('Final counts:', clz_counts)
    print('Done')


def augment_all_in_dir(class_dir, num_outputs):
    input_dir_path = base_dir + "/" + class_dir
    output_dir_path = output_dir + "/" + class_dir
    create_dirs_if_not_found(output_dir_path)
    file_names = os.listdir(input_dir_path)
    aug_idx = 1
    with tqdm(total=num_outputs, desc="Augmenting " + str(class_dir)) as p_bar:
        while len(os.listdir(output_dir_path)) < num_outputs:
            random.shuffle(file_names)
            for file_name in file_names:
                input_path = input_dir_path + "/" + file_name
                output_path = output_dir_path + "/" + str(aug_idx) + "_" + file_name
                img = Image.open(input_path)
                transformed_img = augment_image(img)
                transformed_img.save(output_path)
                if len(os.listdir(output_dir_path)) == num_outputs:
                    return
                p_bar.update(1)
            aug_idx += 1


def augment_image(img):
    # Convert weird half transparent background to completely transparent
    arr = np.array(img)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j][3] == 63:
                arr[i, j] = (0, 0, 0, 0)
    img = Image.fromarray(arr)

    # Create and apply transform
    transform = transforms.RandomAffine(
        degrees=[-180, 180],
    )
    transformed_img = transform(img)

    return transformed_img


if __name__ == "__main__":
    augment_dataset()
