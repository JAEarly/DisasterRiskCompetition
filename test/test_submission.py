import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import log_loss
from tqdm import tqdm

import features
import models
from features import FeatureDatasets, DatasetType
from evaluation.evaluate import run_evaluation
from evaluation.submit import create_from_model
import unittest

SUBMISSION_FOLDER = "./test/resources"
SUBMISSION_FORMAT_PATH = "./data/raw/submission_format.csv"
CORRECT_SUBMISSION_PATH = "./test/resources/correct_comp_submission.csv"


class TestSubmission(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_extractor = features.ResNetCustom()
        self.datasets = FeatureDatasets(self.feature_extractor)
        self.model = models.NNModel(
            models.LinearNN,
            self.feature_extractor.feature_size,
            state_dict_path="./test/resources/resnet_custom_linearnn_model.pth",
            eval_mode=True,
        )
        self.test_submission_filename = "latest_test_submission.csv"

    def test_log_loss(self):
        print("Running log loss test \n")

        test_ll = self.write_test_submission(
            self.test_submission_filename, self.model
        )
        submission_ll = self.evaluate_test_submission(
            self.test_submission_filename, self.feature_extractor
        )
        self.assertAlmostEqual(test_ll, submission_ll)

    def test_submission_output(self):
        print("\nRunning submission output test \n")
        latest_path = "./test/resources/latest_comp_submission.csv"

        create_from_model(
            self.model,
            self.datasets.get_loader(DatasetType.Competition),
            self.feature_extractor.name,
            filepath=latest_path,
        )

        latest_submission = pd.read_csv(latest_path, index_col="id")
        correct_submission = pd.read_csv(CORRECT_SUBMISSION_PATH, index_col="id")

        self.assertEqual(len(latest_submission), len(correct_submission))
        for i in range(len(latest_submission)):
            self.assertEqual(latest_submission.iloc[i].tolist(), correct_submission.iloc[i].tolist())

    def write_test_submission(self, filename, model):
        test_dataset = self.datasets.get_dataset(DatasetType.Test)

        ids = []
        competition_labels = []

        for i in tqdm(range(len(test_dataset)), desc="Predicting test dataset"):
            feature, _, file_id = test_dataset.getitem_filename(i)

            y_output = model.predict(feature)
            if model.apply_softmax:
                y_probas = torch.softmax(y_output, 0)
            else:
                y_probas = y_output
            competition_labels.append(y_probas.cpu().detach().numpy())
            ids.append(file_id)

        # Get labels distribution
        counts = [0] * 5
        for label in competition_labels:
            counts[int(np.argmax(label))] += 1
        print("Class counts:", counts)
        print("Normalised:", ["{:.3f}".format(x / sum(counts)) for x in counts])

        # Write to csv file
        submission = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col="id")
        submission = submission.iloc[0:0]

        # Populate with new submission values
        for label_id, label in tqdm(
            zip(ids, competition_labels), desc="Generating submission", total=len(ids)
        ):
            submission.loc[label_id] = label.tolist()

        # Write submission to csv.
        if not os.path.exists(SUBMISSION_FOLDER):
            os.makedirs(SUBMISSION_FOLDER)
        filepath = os.path.join(SUBMISSION_FOLDER, filename)
        print("Writing submission to", filepath)
        submission.to_csv(filepath)

        test_ll = run_evaluation(self.datasets, model, verbose=False)[-1]
        return test_ll

    @staticmethod
    def evaluate_test_submission(filename, feature_extractor):
        filepath = os.path.join(SUBMISSION_FOLDER, filename)
        submission = pd.read_csv(filepath, index_col="id")
        datasets = FeatureDatasets(feature_extractor)
        test_dataset = datasets.get_dataset(DatasetType.Test)

        y_true = []
        y_pred = []
        for i in tqdm(range(len(test_dataset)), desc="Evaluating test submission"):
            _, label, file_id = test_dataset.getitem_filename(i)
            y_pred.append(submission.loc[file_id].tolist())
            y_true.append(label)

        submission_ll = log_loss(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        return submission_ll


if __name__ == "__main__":
    unittest.main()
