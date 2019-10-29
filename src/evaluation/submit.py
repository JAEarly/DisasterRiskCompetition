import os

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from features.competition_dataset import CompetitionDataset
from utils import create_timestamp_str

SUBMISSION_FOLDER = './submissions'
SUBMISSION_FORMAT_PATH = './data/raw/submission_format.csv'
COMP_DATASET_DIR = "./data/processed/competition"
BATCH_SIZE = 8


def create_submission_from_model(model: models.Model) -> None:
    competition_dataset = CompetitionDataset(COMP_DATASET_DIR, transform=model.get_transform())
    dataloader = DataLoader(competition_dataset, batch_size=BATCH_SIZE, shuffle=False)

    competition_labels = []
    for batch in tqdm(dataloader, desc="Predicting competition dataset"):
        competition_labels.extend(model.predict_batch(batch))

    # Get correct submission format from example submission file
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col='id')
    # Add test labels for this submission
    submission.loc[:, 0:5] = competition_labels

    # Actually write to csv file
    if not os.path.exists(SUBMISSION_FOLDER):
        os.makedirs(SUBMISSION_FOLDER)
    filename = model.name + "_" + create_timestamp_str() + ".csv"
    submission.to_csv(os.path.join(SUBMISSION_FOLDER, filename))


if __name__ == "__main__":
    _model = models.AlexNetSoftmaxModel(state_dict_path="./models/alexnet_softmax_2019-10-29_13:51:45.pth", eval_mode=True)
    create_submission_from_model(_model)
