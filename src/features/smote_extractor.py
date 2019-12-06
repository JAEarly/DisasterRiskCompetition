"""Extension of feature extraction that uses smote balancing."""

import csv
from collections import Counter

import os
from abc import ABC
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

from features import FeatureExtractor, FeatureDatasets, DatasetType
from utils import create_dirs_if_not_found
import torch
from enum import Enum


class SmoteType(Enum):

    Normal = 1
    Borderline = 2
    Svm = 3
    Adasyn = 4


def smote_type_to_name(smote_type: SmoteType):
    if smote_type == SmoteType.Normal:
        return "smote"
    if smote_type == SmoteType.Borderline:
        return "smote_borderline"
    if smote_type == SmoteType.Svm:
        return "smote_svm"
    if smote_type == SmoteType.Adasyn:
        return "adasyn"
    raise NotImplementedError("No name found for ", smote_type)


def smote_type_to_method(smote_type: SmoteType):
    if smote_type == SmoteType.Normal:
        return SMOTE
    if smote_type == SmoteType.Borderline:
        return BorderlineSMOTE
    if smote_type == SmoteType.Svm:
        return SVMSMOTE
    if smote_type == SmoteType.Adasyn:
        return ADASYN
    raise NotImplementedError("No method found for ", smote_type)


class SmoteExtractor(FeatureExtractor, ABC):
    """Feature extractor using smote balancing."""

    def __init__(self, base_feature_extractor: FeatureExtractor, smote_type: SmoteType = SmoteType.Normal):
        super().__init__(base_feature_extractor.name + "_" + smote_type_to_name(smote_type))
        self.base_feature_extractor = base_feature_extractor
        self.feature_datasets = FeatureDatasets(self.base_feature_extractor, oversample_validation=False)
        self.smote_type = smote_type
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
            not os.path.exists(training_dir) or current_size < expected_size
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
            train_features, train_labels = self.feature_datasets.get_features_and_labels(
                DatasetType.Train
            )
            val_features, val_labels = self.feature_datasets.get_features_and_labels(
                DatasetType.Validation
            )
            # test_features, test_labels = self.feature_datasets.get_features_and_labels(
            #     DatasetType.Test
            # )
            all_features = torch.cat([train_features, val_features]).cpu().detach()
            all_labels = torch.cat([train_labels, val_labels]).cpu().detach()
            print(
                "Original training distribution -", Counter([l.item() for l in train_labels])
            )
            print(
                "Original dataset distribution -", Counter([l.item() for l in all_labels])
            )

            # Run smote
            print("Running smote")
            smt = smote_type_to_method(self.smote_type)()
            # TODO reduce to only training set length?
            all_features, all_labels = smt.fit_resample(all_features, all_labels)
            print("SMOTE dataset distribution -", Counter([l.item() for l in all_labels]))

            print("Saving tensors")
            # Save feature tensors
            i = 0
            filenames = []
            for feature in all_features:
                self._save_tensor(DatasetType.Train, feature, i)
                filenames.append(i)
                i += 1

            print("Saving labels")
            # Save labels file
            labels_filepath = self.get_labels_filepath(DatasetType.Train)
            with open(labels_filepath, "w+") as file:
                csv_writer = csv.writer(file)
                for filename, label in zip(filenames, all_labels):
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
                self.save_dir, self.name, dataset_type.name.lower()
            )
        return self.base_feature_extractor.get_features_dir(dataset_type)

    def get_labels_filepath(self, dataset_type: DatasetType) -> str:
        """
        Get the filepath for the labels file.
        :param dataset_type: Dataset in use (train, test etc.).
        :return: None.
        """
        if dataset_type == DatasetType.Train:
            return os.path.join(
                self.save_dir,
                self.name,
                dataset_type.name.lower() + "_labels.pkl",
            )
        return self.base_feature_extractor.get_labels_filepath(dataset_type)
