"""Script for creating submissions."""

import statistics

import numpy as np
import os
import pandas as pd
import torch
from torchvision import models as tv_models
from tqdm import tqdm

import features
import models
import models.transfers as transfers
from features import FeatureDatasets, ImageDatasets, DatasetType
from utils import create_timestamp_str

SUBMISSION_FOLDER = "./submissions"
SUBMISSION_FORMAT_PATH = "./data/raw/submission_format.csv"
DEFAULT_CONFIDENCE_THRESHOLD = 0.99


def create_from_model(
    model: models.Model,
    competition_loader,
    feature_name: str,
    confidence_threshold=None,
    filepath=None,
) -> None:
    """
    Create a submission.
    :param model: Model to use.
    :param competition_loader: Competition loader to use.
    :param feature_name: Tag for feature extractor used.
    :param confidence_threshold: Threshold for boosting. If None, no boosting is used.
    :return: None.
    """
    # Get predicted labels for competition data
    ids = []
    competition_labels = []
    for batch, batch_ids in tqdm(
        competition_loader, desc="Predicting competition dataset"
    ):
        y_outputs = model.predict_batch(batch)
        if model.apply_softmax:
            y_probas = torch.softmax(y_outputs, 1)
        else:
            y_probas = y_outputs
        competition_labels.extend(y_probas.cpu().detach().numpy())
        ids.extend(batch_ids)

    # Boost if requested
    if confidence_threshold is not None:
        competition_labels = boost_labels(
            competition_labels, confidence_threshold=confidence_threshold
        )

    # Get labels distribution
    counts = [0] * 5
    for label in competition_labels:
        counts[int(np.argmax(label))] += 1
    print("Class counts:", counts)
    print("Normalised:", ["{:.3f}".format(x / sum(counts)) for x in counts])

    # Write to csv file
    if filepath is None:
        if not os.path.exists(SUBMISSION_FOLDER):
            os.makedirs(SUBMISSION_FOLDER)
        filename = feature_name + "_" + model.name + "_" + create_timestamp_str() + ".csv"
        filepath = os.path.join(SUBMISSION_FOLDER, filename)
    _write_submission(ids, competition_labels, filepath)


def boost_existing(filename, competition_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Boost an existing file.
    :param filename: Filename to boost.
    :param competition_threshold: Threshold above which labels will be maxed (i.e. set to 1).
    :return: None
    """
    print("Boosting", filename, "\n")

    # Parse existing submission
    filepath = os.path.join(SUBMISSION_FOLDER, filename)
    submission = pd.read_csv(filepath, index_col="id")
    ids = []
    competition_labels = []
    for label_id, row in submission.iterrows():
        ids.append(label_id)
        competition_labels.append(row.to_numpy().tolist())

    # Boost
    competition_labels = boost_labels(competition_labels, competition_threshold)

    # Write to csv file
    filename = filename[:-4] + "_boosted.csv"
    _write_submission(ids, competition_labels, filename)


def _write_submission(ids, competition_labels, filepath):
    """
    Write a submission to csv.
    :param ids: List of prediction ids.
    :param competition_labels: Labels for prediction ids (order should match ids).
    :param filename: Name of submission file.
    :return: None.
    """
    # Create dataframe from submission format
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col="id")

    # Populate with new submission values
    for label_id, label in tqdm(
        zip(ids, competition_labels), desc="Generating submission", total=len(ids)
    ):
        submission[submission.index.str.startswith(label_id)] = [label]

    # Write submission to csv.
    print("Writing submission to", filepath)
    submission.to_csv(filepath)


def boost_labels(
    competition_labels,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    confirm=True,
    verbose=True,
):
    """
    Boost a set of competition labels. Asks for confirmation.
    :param competition_labels: Labels to boost.
    :param confidence_threshold: Threshold above which labels will be maxed (i.e. set to 1).
    :param confirm: Ask for confirmation from user.
    :return: Boosted labels.
    """
    if verbose:
        # Establish max, avg, min confidences of current labels.
        max_confidences = []
        for label in competition_labels:
            max_confidences.append(max(label))
        print("Max confidence:", max(max_confidences))
        print("Avg confidence:", statistics.mean(max_confidences))
        print("Min confidence:", min(max_confidences))

        # Establish number of labels that will be boosted and ask for user confirmation.
        print("Boosting confidence with threshold", confidence_threshold)
        num_to_boost = sum(x >= confidence_threshold for x in max_confidences)
        print(
            "Will boost",
            str(num_to_boost) + "/" + str(len(competition_labels)),
            "labels",
        )
    if confirm:
        reply = str(input("Continue? (Y/n)")).lower().strip()
        if not (reply == "" or reply[0] == "y" or reply[0] == "Y"):
            print("Aborting")
            exit(0)
        print("Continuing...")

    # Boost labels
    boosted_labels = []
    for label in competition_labels:
        if max(label) >= confidence_threshold:
            boosted_label = np.zeros(5)
            boosted_label[np.argmax(label)] = 1.0
            boosted_labels.append(boosted_label)
        else:
            boosted_labels.append(label)
    return boosted_labels


def _setup_feature_submission():
    """Get required information for a feature based submission."""
    feature_extractor = features.ResNetCustomSMOTE()
    datasets = FeatureDatasets(feature_extractor)

    # model = models.NNModel(
    #     models.LinearNN,
    #     feature_extractor.feature_size,
    #     state_dict_path="./models/grid_search_resnet_custom_linearnn_2/best.pth",
    #     eval_mode=True,
    # )

    model = models.XGBModel(
        model_path="./models/grid_search_resnet_custom_smote_xgb_4/best.pth"
    )

    print("Running submission for", feature_extractor.name, model.name, "\n")
    return model, datasets.get_loader(DatasetType.Competition), feature_extractor.name


def _setup_image_submission():
    """Get required information for an image based submission."""
    datasets = ImageDatasets()

    model = models.PretrainedNNModel(
        tv_models.resnet152,
        transfers.final_layer_alteration_resnet,
        state_dict_path="./models/grid_search_resnet_custom/best.pth",
        eval_mode=True,
    )
    print("Running submission for", model.name, "\n")
    return model, datasets.get_loader(DatasetType.Competition), "custom"


if __name__ == "__main__":
    _model, _competition_loader, _feature_name = _setup_feature_submission()
    # _model, _competition_dataset, _feature_name = setup_image_submission()

    create_from_model(_model, _competition_loader, _feature_name)

    # boost_existing("resnet_custom_biggernn_2019-11-28_18:27:37.csv", competition_threshold=0.99993896484375)
