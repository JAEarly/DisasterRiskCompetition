from abc import ABC, abstractmethod
from torch import nn
from tqdm import tqdm
import os
import pickle
from torchvision import datasets
from torch.utils import data
from utils import create_dirs_if_not_found
import torch


class FeatureExtractor(ABC):

    train_dir = "./data/processed/train"
    validation_dir = "./data/processed/validation"
    test_dir = "./data/processed/test"
    save_dir = "./models/features/"
    batch_size = 8

    def __init__(self, name):
        self.name = name
        self.extractor_model = self.setup_model()

        # TODO create dataset class
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.get_transform())
        self.validation_dataset = datasets.ImageFolder(self.validation_dir, transform=self.get_transform())
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.get_transform())

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def get_transform(self):
        pass

    def extract(self, dataset_loader, feature_type):
        # TODO don't pass in dataset loader
        features, labels = self.load(feature_type)
        if features is None or labels is None:
            features = []
            labels = []
            for batch, batch_labels in tqdm(dataset_loader, desc='Extracting features  - ' + feature_type):
                features.extend(self.extractor_model(batch))
                labels.extend(batch_labels.cpu().detach().numpy())
            features = torch.stack(features)
            features = features.cpu().detach()
            self.save(features, labels, feature_type)
        return features, labels

    def save(self, features, labels, feature_type):
        create_dirs_if_not_found(self.save_dir)
        features_filepath = self._get_features_filepath(feature_type)
        labels_filepath = self._get_labels_filepath(feature_type)
        with open(features_filepath, "wb") as file:
            pickle.dump(features, file)
        with open(labels_filepath, "wb") as file:
            pickle.dump(labels, file)

    def load(self, feature_type):
        features_filepath = self._get_features_filepath(feature_type)
        labels_filepath = self._get_labels_filepath(feature_type)
        if os.path.exists(features_filepath) and os.path.exists(labels_filepath):
            with open(features_filepath, "rb") as file:
                features = pickle.load(file)
            with open(labels_filepath, "rb") as file:
                labels = pickle.load(file)
            return features, labels
        return None, None

    def _get_features_filepath(self, feature_type):
        return os.path.join(self.save_dir, self.name + "_" + feature_type + "_features.pkl")

    def _get_labels_filepath(self, feature_type):
        return os.path.join(self.save_dir, self.name + "_" + feature_type + "_labels.pkl")


class IdentityLayer(nn.Module):
    """
    Identity layer that just passes input through to output.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
