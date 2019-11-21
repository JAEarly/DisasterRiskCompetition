"""
Contains the abstract and implemented classes for a datasets wrapper.

A datasets class contains train, validation and test datasets (all labelled data).
These datasets are either backed by images (raw data) or features (extracted from images).
"""

import pickle
from enum import Enum
from typing import Tuple

import os
import torch
from abc import ABC, abstractmethod
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import datasets as torch_datasets
from torchvision.transforms import transforms


class DatasetType(Enum):
    """Enum for dataset types."""

    Train = 1
    Validation = 2
    Test = 3
    Competition = 4


class BalanceMethod(Enum):
    """Enum for dataset balancing methods."""

    NoSample = 0
    UnderSample = 1
    AvgSample = 2
    OverSample = 3


class Datasets(ABC):
    """Abstract base class for datasets."""

    def __init__(self, batch_size=8, balance_method=BalanceMethod.NoSample):
        self.batch_size = batch_size

        # Create datasets
        (
            self.train_dataset,
            self.validation_dataset,
            self.test_dataset,
        ) = self.create_datasets(balance_method)

        # Create dataloaders from datasets.
        self.train_loader = data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.validation_loader = data.DataLoader(
            self.validation_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    @abstractmethod
    def create_datasets(
        self, balance_method: BalanceMethod
    ) -> Tuple[data.Dataset, data.Dataset, data.Dataset]:
        """
        Create the train, validation and test datasets.
        :param balance_method: Method for balancing the training dataset.
        :return: Train, validation and test datasets as a tuple.
        """

    def get_dataset(self, dataset_type: DatasetType) -> data.Dataset:
        """
        Get a specific dataset.
        :param dataset_type: The type of dataset to return (train, validation etc.)
        :return: Dataset (if found).
        """
        if dataset_type == DatasetType.Train:
            return self.train_dataset
        if dataset_type == DatasetType.Validation:
            return self.validation_dataset
        if dataset_type == DatasetType.Test:
            return self.test_dataset
        raise IndexError("Could not find dataset for type " + dataset_type.name)

    def get_loader(self, dataset_type: DatasetType) -> data.DataLoader:
        """
        Get a specific dataloader.
        :param dataset_type: The type of dataloader to return (train, validation etc.)
        :return: Dataloader (if found).
        """
        if dataset_type == DatasetType.Train:
            return self.train_loader
        if dataset_type == DatasetType.Validation:
            return self.validation_loader
        if dataset_type == DatasetType.Test:
            return self.test_loader
        raise IndexError("Could not find data loader for type " + dataset_type.name)


class ImageDatasets(Datasets):
    """Implementation of Datasets backed with images."""

    train_dir = "./data/processed/train"
    validation_dir = "./data/processed/validation"
    test_dir = "./data/processed/test"

    def __init__(self, transform: transforms.Compose = None):
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        super().__init__()

    def create_datasets(self, balance_method: BalanceMethod):
        # Create image datasets using folder paths and given transform.
        train_dataset = torch_datasets.ImageFolder(
            self.train_dir, transform=self.transform
        )
        validation_dataset = torch_datasets.ImageFolder(
            self.validation_dir, transform=self.transform
        )
        test_dataset = torch_datasets.ImageFolder(
            self.test_dir, transform=self.transform
        )

        return train_dataset, validation_dataset, test_dataset


class FeatureDataset(Dataset):
    """Feature dataset implementation. Uses pickled tensors for each feature vector."""

    def __init__(
        self, features_dir, labels_path, balance_method=BalanceMethod.NoSample
    ):
        super().__init__()
        self.data_dir = features_dir
        self.filenames = os.listdir(self.data_dir)
        with open(labels_path, "rb") as file:
            self.labels = pickle.load(file)
        if balance_method == BalanceMethod.UnderSample:
            self._undersample()
        elif balance_method == BalanceMethod.AvgSample:
            self._avgsample()
        elif balance_method == BalanceMethod.OverSample:
            self._oversample()

    def _undersample(self):
        data_dist = [0] * 5
        for label in self.labels:
            data_dist[label] += 1
        min_class_size = min(data_dist)
        self._sample_to_target(min_class_size)

    def _avgsample(self):
        data_dist = [0] * 5
        for label in self.labels:
            data_dist[label] += 1
        avg_class_size = int(sum(data_dist) / 5)
        self._sample_to_target(avg_class_size)

    def _oversample(self):
        data_dist = [0] * 5
        for label in self.labels:
            data_dist[label] += 1
        max_class_size = max(data_dist)
        self._sample_to_target(max_class_size)

    def _sample_to_target(self, target):
        data_dist = [0] * 5
        reduced_filenames = []
        reduced_labels = []
        while len(reduced_filenames) < target * 5:
            for filename, label in zip(self.filenames, self.labels):
                if data_dist[label] < target:
                    reduced_filenames.append(filename)
                    reduced_labels.append(label)
                    data_dist[label] += 1
        self.filenames = reduced_filenames
        self.labels = reduced_labels

    def _load_vector(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "rb") as file:
            feature = pickle.load(file)
            if len(feature.shape) == 2:
                feature = feature[0]
        return feature

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return self._load_vector(self.filenames[index]), self.labels[index]


class FeatureDatasets(Datasets):
    """Implementation of Datasets back with a feature extractor."""

    def __init__(self, feature_extractor, balance_method=BalanceMethod.NoSample):
        # Ensure features are extracted
        self.feature_extractor = feature_extractor
        self.feature_extractor.extract(DatasetType.Train)
        self.feature_extractor.extract(DatasetType.Validation)
        self.feature_extractor.extract(DatasetType.Test)
        super().__init__(balance_method=balance_method)

    def create_datasets(self, balance_method):
        # Create feature datasets
        train_dataset = FeatureDataset(
            self.feature_extractor.get_features_dir(DatasetType.Train),
            self.feature_extractor.get_labels_filepath(DatasetType.Train),
            balance_method,
        )
        validation_dataset = FeatureDataset(
            self.feature_extractor.get_features_dir(DatasetType.Validation),
            self.feature_extractor.get_labels_filepath(DatasetType.Validation),
            balance_method=BalanceMethod.NoSample,
        )
        test_dataset = FeatureDataset(
            self.feature_extractor.get_features_dir(DatasetType.Test),
            self.feature_extractor.get_labels_filepath(DatasetType.Test),
            balance_method=BalanceMethod.NoSample,
        )
        return train_dataset, validation_dataset, test_dataset

    def get_features_and_labels(
        self, dataset_type: DatasetType
    ) -> (torch.Tensor, torch.Tensor):
        """
        Get all the features and labels for a dataset.
        :param dataset_type: Dataset to fetch labels for.
        :return: Tensor of features and tensor of labels.
        """
        features = []
        labels = []
        for batch, batch_labels in self.get_loader(dataset_type):
            features.extend(batch)
            labels.extend(batch_labels)
        features = torch.stack(features)
        labels = torch.tensor(labels)
        return features, labels
