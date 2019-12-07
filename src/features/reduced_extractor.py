import csv
import pickle

import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from features import FeatureExtractor, FeatureDatasets, DatasetType
from utils import create_dirs_if_not_found
import torch
from abc import ABC, abstractmethod


class ReducedExtractor(FeatureExtractor, ABC):

    def __init__(
        self,
        name,
        base_feature_extractor,
        num_components,
    ):
        self.feature_datasets = FeatureDatasets(base_feature_extractor)
        self.num_components = num_components
        super().__init__(
            name,
            save_dir=base_feature_extractor.save_dir,
            train_dir=base_feature_extractor.train_dir,
        )

    def setup_model(self):
        model_save_path = self.get_model_save_path()
        if os.path.exists(model_save_path):
            with open(model_save_path, "rb") as file:
                reduction_model = pickle.load(file)
            print("Reduction model found")
        else:
            print("Setting up reduction model")
            create_dirs_if_not_found(os.path.join(self.save_dir, self.name))
            reduction_model = self.create_reduction_model()
            print("Saving reduction model")
            with open(model_save_path, "wb") as file:
                pickle.dump(reduction_model, file)
        return reduction_model, self.num_components

    @abstractmethod
    def create_reduction_model(self):
        pass

    def extract(self, dataset_type: DatasetType):
        features_dir = self.get_features_dir(dataset_type)
        features_dataset = self.feature_datasets.get_dataset(dataset_type)
        if not os.path.exists(features_dir) or len(os.listdir(features_dir)) != len(
            features_dataset
        ):
            print("Extracting", dataset_type, "for", self.name)
            create_dirs_if_not_found(features_dir)
            reduced_features, labels = self.extract_for_dataset(dataset_type)

            print("Saving tensors")
            # Save feature tensors
            i = 0
            filenames = []
            for reduced_feature in reduced_features:
                reduced_feature = torch.tensor(reduced_feature)
                reduced_feature = reduced_feature.float()
                self._save_tensor(dataset_type, reduced_feature, i)
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

    def extract_for_dataset(self, dataset_type: DatasetType):
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
        reduced_features = self.extractor_model.transform(features)
        return reduced_features, labels

    def get_model_save_path(self):
        return os.path.join(self.save_dir, self.name, "reduction_model.pkl")


class ReducedBasicExtractor(ReducedExtractor):

    def __init__(
        self,
        base_feature_extractor,
        num_components,
    ):
        super().__init__(
            base_feature_extractor.name + "_reduced_" + str(num_components),
            base_feature_extractor,
            num_components
        )

    def create_reduction_model(self):
        print("Getting training features")
        training_features = (
            self.feature_datasets.get_features(DatasetType.Train).cpu().detach()
        )
        reduction_model = PCA(n_components=self.num_components)
        print("Scaling data")
        training_features = StandardScaler().fit_transform(training_features)
        print("Running " + str(reduction_model))
        reduction_model.fit_transform(training_features)
        return reduction_model
