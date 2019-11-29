"""Extension of feature extraction that uses smote balancing."""

import csv
from collections import Counter

import os
from abc import ABC
from imblearn.over_sampling import SMOTE

from features import FeatureExtractor, FeatureDatasets, DatasetType
from utils import create_dirs_if_not_found


class SmoteExtractor(FeatureExtractor, ABC):
    """Feature extractor using smote balancing."""

    def __init__(self, base_feature_extractor: FeatureExtractor):
        super().__init__(base_feature_extractor.name)
        self.base_feature_extractor = base_feature_extractor
        self.feature_datasets = FeatureDatasets(self.base_feature_extractor)
        labels = self.feature_datasets.get_dataset(
            DatasetType.Train
        ).path2label.values()

        training_dir = self.get_features_dir(DatasetType.Train)
        create_dirs_if_not_found(training_dir)

        features_dist = Counter(labels).values()
        biggest_class = max(features_dist)
        current_size = len(os.listdir(training_dir))
        expected_size = biggest_class * len(features_dist)

        # Check if extraction has already happened
        self.extraction_required = (
            not os.path.exists(training_dir) or current_size != expected_size
        )

    def extract(self, dataset_type: DatasetType) -> None:
        if dataset_type == DatasetType.Train:
            self.extract_smote()
        else:
            super().extract(dataset_type)

    def extract_smote(self) -> None:
        """
        Run extraction with SMOTE balancing.
        :return: None.
        """
        if self.extraction_required:
            print("Running smote extraction for", self.name)
            features, labels = self.feature_datasets.get_features_and_labels(
                DatasetType.Train
            )

            print(
                "Original dataset distribution -", Counter([l.item() for l in labels])
            )
            features = features.cpu().detach()
            labels = labels.cpu().detach()

            # Run smote
            print("Running smote")
            smt = SMOTE()
            features, labels = smt.fit_sample(features, labels)
            print("SMOTE dataset distribution -", Counter([l.item() for l in labels]))

            print("Saving tensors")
            # Save feature tensors
            i = 0
            filenames = []
            for feature in features:
                self._save_tensor(DatasetType.Train, feature, i)
                filenames.append(i)
                i += 1

            print("Saving labels")
            # Save labels file
            labels_filepath = self.get_labels_filepath(DatasetType.Train)
            with open(labels_filepath, "w+") as file:
                csv_writer = csv.writer(file)
                for filename, label in zip(filenames, labels):
                    csv_writer.writerow([filename, label])

            print("Done")

    def get_features_dir(self, dataset_type: DatasetType) -> str:
        """
        Get the directory for a set of features.
        :param dataset_type: Dataset in use (train, test etc.).
        :return: None.
        """
        if dataset_type == DatasetType.Train:
            return os.path.join(
                self.save_dir, self.name + "_smote", dataset_type.name.lower()
            )
        return super().get_features_dir(dataset_type)

    def get_labels_filepath(self, dataset_type: DatasetType) -> str:
        """
        Get the filepath for the labels file.
        :param dataset_type: Dataset in use (train, test etc.).
        :return: None.
        """
        if dataset_type == DatasetType.Train:
            return os.path.join(
                self.save_dir,
                self.name + "_smote",
                dataset_type.name.lower() + "_labels.pkl",
            )
        return super().get_labels_filepath(dataset_type)
