import os

import pandas as pd
from tqdm import tqdm

from features.competition_dataset import CompetitionDataset
from models import Model, BaselineModel

SUBMISSION_FOLDER = './submissions'
SUBMISSION_FORMAT_PATH = './data/raw/submission_format.csv'
COMP_DATASET_DIR = "./data/processed/comp"


def create_submission_from_model(model: Model, submission_name: str) -> None:
    competition_dataset = CompetitionDataset(COMP_DATASET_DIR, transform=model.get_transform())
    competition_labels = []
    for x in tqdm(competition_dataset, desc="Predicting competition dataset"):
        label = model.predict(x)
        competition_labels.append(label)

    # Get correct submission format from example submission file
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col='id')
    # Add test labels for this submission
    submission.loc[:, 0:5] = competition_labels

    # Actually write to csv file
    if not os.path.exists(SUBMISSION_FOLDER):
        os.makedirs(SUBMISSION_FOLDER)
    submission.to_csv(os.path.join(SUBMISSION_FOLDER, submission_name + ".csv"))


if __name__ == "__main__":
    create_submission_from_model(BaselineModel(), "baseline")
    #create_submission_from_model(AlexNetModel(state_dict_path="./models/alexnet_transfer", eval_mode=True), "alexnet")
