import os
import pickle
from abc import ABC, abstractmethod

import torch
from torch import nn
from tqdm import tqdm

from features import ImageDatasets, DatasetType
from utils import create_dirs_if_not_found


class FeatureExtractor(ABC):

    save_dir = "./models/features/"

    def __init__(self, name):
        self.name = name
        self.extractor_model = self.setup_model()
        self.image_datasets = ImageDatasets(self.get_transform())

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def get_transform(self):
        pass

    def extract(self, dataset_type: DatasetType):
        features, labels = self.load(dataset_type)
        if features is None or labels is None:
            features = []
            labels = []
            dataset_loader = self.image_datasets.get_loader(dataset_type)
            for batch, batch_labels in tqdm(dataset_loader, desc='Extracting features  - ' + dataset_type.name):
                features.extend(self.extractor_model(batch))
                labels.extend(batch_labels.cpu().detach().numpy())
            features = torch.stack(features)
            features = features.cpu().detach()
            self.save(features, labels, dataset_type)
        return features, labels

    def save(self, features, labels, dataset_type: DatasetType):
        create_dirs_if_not_found(self.save_dir)
        features_filepath = self._get_features_filepath(dataset_type)
        labels_filepath = self._get_labels_filepath(dataset_type)
        with open(features_filepath, "wb") as file:
            pickle.dump(features, file)
        with open(labels_filepath, "wb") as file:
            pickle.dump(labels, file)

    def load(self, dataset_type: DatasetType):
        features_filepath = self._get_features_filepath(dataset_type)
        labels_filepath = self._get_labels_filepath(dataset_type)
        if os.path.exists(features_filepath) and os.path.exists(labels_filepath):
            with open(features_filepath, "rb") as file:
                features = pickle.load(file)
            with open(labels_filepath, "rb") as file:
                labels = pickle.load(file)
            return features, labels
        return None, None

    def _get_features_filepath(self, dataset_type: DatasetType):
        return os.path.join(self.save_dir, self.name + "_" + dataset_type.name + "_features.pkl")

    def _get_labels_filepath(self, dataset_type: DatasetType):
        return os.path.join(self.save_dir, self.name + "_" + dataset_type.name + "_labels.pkl")


class IdentityLayer(nn.Module):
    """
    Identity layer that just passes input through to output.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
