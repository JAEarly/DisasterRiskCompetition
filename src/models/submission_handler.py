import pandas as pd
import os

SUBMISSION_FOLDER = '../submissions'
SUBMISSION_FORMAT_PATH = '../data/submission_format.csv'


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
