"""Script for creating submissions."""

import os

import pandas as pd
import torch
from tqdm import tqdm

import features
import models
from features import CompetitionFeatureDataset
from utils import create_timestamp_str

SUBMISSION_FOLDER = "./submissions"
SUBMISSION_FORMAT_PATH = "./data/raw/submission_format.csv"


def create_submission(
    model: models.Model, competition_dataset: CompetitionFeatureDataset
) -> None:
    """
    Create a submission.
    :param model: Model to use.
    :param competition_dataset: Competition dataset to use.
    :return: None.
    """
    # Get predicted labels for competition data
    competition_labels = []
    for batch in tqdm(
        competition_dataset.data_loader, desc="Predicting competition dataset"
    ):
        y_outputs = model.predict_batch(batch)
        y_probas = torch.softmax(y_outputs, 1)
        competition_labels.extend(y_probas.cpu().detach().numpy())

    # Get correct submission format from example submission file
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col="id")

    # Add test labels for this submission
    submission.loc[:, 0:5] = competition_labels

    # Actually write to csv file
    if not os.path.exists(SUBMISSION_FOLDER):
        os.makedirs(SUBMISSION_FOLDER)
    filename = (
        competition_dataset.feature_extractor.name
        + "_"
        + model.name
        + "_"
        + create_timestamp_str()
        + ".csv"
    )
    submission.to_csv(os.path.join(SUBMISSION_FOLDER, filename))


if __name__ == "__main__":
    _feature_extractor = features.AlexNet()
    _model = models.NNModel(
        models.BiggerNN,
        _feature_extractor.feature_size,
        state_dict_path="./models/alexnet_biggernn_2019-11-11_20:07:44.pth",
        eval_mode=True,
    )
    create_submission(_model, CompetitionFeatureDataset(_feature_extractor))
