import pandas as pd
import os
import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial
from torchvision import models, transforms, datasets
import src.features.alexnet_feature_extractor as al_fe
from torch.utils import data
import os
from PIL import Image
import numpy as np

SUBMISSION_FOLDER = '../submissions'
SUBMISSION_FORMAT_PATH = '../data/submission_format.csv'

data_dir = "./data/processed/test"
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()])


class TestDataset(data.Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.id_list = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, self.id_list[index]))
        if self.transform is not None:
            image = self.transform(image)
        image = np.transpose(image, (1, 2, 0))
        return image


def create_submission_from_model(model):
    test_dataset = TestDataset(data_dir, transform=transform)
    for x in test_dataset:
        print(x.shape)
        print(model(x.unsqueeze(0)))
        break


def create_submission(test_labels, submission_name) -> None:
    """
    Write the test labels to a submission file.

    :param test_labels: Class probabilities for each datapoint in the test set.
    :param submission_name: Name of submission (not file name)
    """
    # Get correct submission format from example submission file
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col='id')
    # Add test labels for this submission
    submission.loc[:, 0:5] = test_labels

    # Actually write to csv file
    if not os.path.exists(SUBMISSION_FOLDER):
        os.makedirs(SUBMISSION_FOLDER)
    submission.to_csv(os.path.join(SUBMISSION_FOLDER, submission_name + ".csv"))


if __name__ == "__main__":
    al_model = al_fe.get_model()
    al_model.load_state_dict(torch.load("./models/alexnet_transfer"))
    al_model.eval()
    create_submission_from_model(al_model)
