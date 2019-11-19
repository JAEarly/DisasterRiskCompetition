"""Script for creating submissions."""

import os
import pandas as pd
import torch
from torchvision import models as tv_models
from tqdm import tqdm

import features
import models
import models.cnn_model as cnn_models
from features import (
    CompetitionDataset,
    CompetitionFeatureDataset,
    CompetitionImageDataset,
)
from utils import create_timestamp_str

SUBMISSION_FOLDER = "./submissions"
SUBMISSION_FORMAT_PATH = "./data/raw/submission_format.csv"


def create_submission(
    model: models.Model, competition_dataset: CompetitionDataset, feature_name: str
) -> None:
    """
    Create a submission.
    :param model: Model to use.
    :param competition_dataset: Competition dataset to use.
    :return: None.
    """
    # Get predicted labels for competition data
    ids = []
    competition_labels = []
    for batch_ids, batch in tqdm(
        competition_dataset.data_loader, desc="Predicting competition dataset"
    ):
        y_outputs = model.predict_batch(batch)
        y_probas = torch.softmax(y_outputs, 1)
        competition_labels.extend(y_probas.cpu().detach().numpy())
        ids.extend(batch_ids)

    # Get correct submission format from example submission file
    print("Creating output")
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col="id")
    for label_id, label in zip(ids, competition_labels):
        label = label.tolist()
        submission[submission.index.str.startswith(label_id)] = [label]

    # Actually write to csv file
    print("Write to file")
    if not os.path.exists(SUBMISSION_FOLDER):
        os.makedirs(SUBMISSION_FOLDER)
    filename = feature_name + "_" + model.name + "_" + create_timestamp_str() + "_2.csv"
    submission.to_csv(os.path.join(SUBMISSION_FOLDER, filename))


def setup_feature_submission():
    feature_extractor = features.ResNet()
    model = models.NNModel(
        models.BiggerNN,
        feature_extractor.feature_size,
        state_dict_path=(
            "./models/grid_search_resnet_biggernn/resnet_biggernn_best.pth"
        ),
        eval_mode=True,
    )
    print("Running submission for", feature_extractor.name, model.name)
    return model, CompetitionFeatureDataset(feature_extractor), feature_extractor.name


def setup_image_submission():
    model = models.PretrainedNNModel(
        tv_models.alexnet,
        cnn_models.final_layer_alteration_alexnet,
        state_dict_path="./models/grid_search_alexnet_cnn/best.pth",
        eval_mode=True,
    )
    print("Running submission for", model.name)
    return model, CompetitionImageDataset(), "images"


if __name__ == "__main__":
    # TODO feature submission is broken
    # _model, _competition_dataset, _feature_name = setup_feature_submission()
    _model, _competition_dataset, _feature_name = setup_image_submission()
    create_submission(_model, _competition_dataset, _feature_name)
