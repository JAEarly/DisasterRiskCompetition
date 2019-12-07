"""Extension of feature extraction that uses smote balancing."""

import csv

import os
from abc import ABC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from features import FeatureExtractor, FeatureDatasets, DatasetType
from utils import create_dirs_if_not_found
import pickle


class ReducedExtractor:

    def __init__(self, feature_extractor, num_components, save_dir="./models/features/"):
        self.name = feature_extractor.name + "_reduced_" + str(num_components)
        self.save_dir = save_dir
        self.feature_datasets = FeatureDatasets(feature_extractor)
        self.num_components = num_components
        self.reduction_model = self.setup_model()

    def setup_model(self):
        model_save_path = self.get_model_save_path()
        if os.path.exists(model_save_path):
            with open(model_save_path, "rb") as file:
                reduction_model = pickle.load(file)
            print("Reduction model found")
        else:
            print("Setting up reduction model")
            print("Getting training features")
            training_features = self.feature_datasets.get_features(DatasetType.Train).cpu().detach()
            reduction_model = PCA(n_components=self.num_components)
            print("Scaling data")
            training_features = StandardScaler().fit_transform(training_features)
            print("Running " + str(reduction_model))
            reduction_model.fit_transform(training_features)
            print("Saving reduction model")
            with open(model_save_path, "wb") as file:
                pickle.dump(reduction_model, file)
        return reduction_model

    def extract(self, dataset_type: DatasetType):
        features_dir = self.get_features_dir(dataset_type)
        features_dataset = self.feature_datasets.get_dataset(dataset_type)
        if not os.path.exists(features_dir) or len(os.listdir(features_dir)) != len(
            features_dataset
        ):
            print("Extracting", dataset_type, "for", self.name)
            create_dirs_if_not_found(features_dir)
            if dataset_type == DatasetType.Competition:
                features = self.feature_datasets.get_features(DatasetType.Competition)
                labels = None
            else:
                features, labels = self.feature_datasets.get_features_and_labels(
                    dataset_type
                )
                labels = labels.cpu().detach().numpy()
            features = features.cpu().detach()
            print("Running reduction")
            reduced_features = self.reduction_model.transform(features)

            print("Saving tensors")
            # Save feature tensors
            i = 0
            filenames = []
            for reduced_feature in reduced_features:
                self.save_tensor(dataset_type, reduced_feature, i)
                filenames.append(i)
                i += 1

            if labels is not None:
                print("Saving labels")
                # Save labels file
                labels_filepath = self.get_labels_filepath(dataset_type)
                with open(labels_filepath, "w+") as file:
                    csv_writer = csv.writer(file)
                    for filename, label in zip(filenames, labels):
                        csv_writer.writerow([filename, label])

    def get_model_save_path(self):
        return os.path.join(self.save_dir, self.name, "reduction_model.pkl")

    def get_features_dir(self, dataset_type: DatasetType) -> str:
        return os.path.join(self.save_dir, self.name, dataset_type.name.lower())

    def get_labels_filepath(self, dataset_type: DatasetType) -> str:
        return os.path.join(
            self.save_dir, self.name, dataset_type.name.lower() + "_labels.csv"
        )

    def save_tensor(self, dataset_type, tensor, file_id) -> None:
        feature_dir = self.get_features_dir(dataset_type)
        path = os.path.join(feature_dir, str(file_id) + ".pkl")
        with open(path, "wb") as file:
            pickle.dump(tensor, file)
