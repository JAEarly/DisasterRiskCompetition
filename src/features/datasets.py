"""
Contains the abstract and implemented classes for a datasets wrapper.

A datasets class contains train, validation and test datasets (all labelled data).
These datasets are either backed by images (raw data) or features (extracted from images).
"""

import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import torch
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


class Datasets(ABC):
    """Abstract base class for datasets."""

    def __init__(self, batch_size=8):
        self.batch_size = batch_size

        # Create datasets
        (
            self.train_dataset,
            self.validation_dataset,
            self.test_dataset,
        ) = self.create_datasets()

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
    def create_datasets(self) -> Tuple[data.Dataset, data.Dataset, data.Dataset]:
        """
        Create the train, validation and test datasets.
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

    def __init__(self, transform: transforms.Compose):
        self.transform = transform
        super().__init__()

    def create_datasets(self):
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

    def __init__(self, features_dir, labels_path):
        super().__init__()
        self.data_dir = features_dir
        self.filenames = os.listdir(self.data_dir)
        with open(labels_path, "rb") as file:
            self.labels = pickle.load(file)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_dir, self.filenames[index])
        with open(filepath, "rb") as file:
            feature = pickle.load(file)[0]
        return feature, self.labels[index]


class FeatureDatasets(Datasets):
    """Implementation of Datasets back with a feature extractor."""

    def __init__(self, feature_extractor):
        # Ensure features are extracted
        self.feature_extractor = feature_extractor
        self.feature_extractor.extract(DatasetType.Train)
        self.feature_extractor.extract(DatasetType.Validation)
        self.feature_extractor.extract(DatasetType.Test)
        super().__init__()

    def create_datasets(self):
        # Create feature datasets
        train_dataset = FeatureDataset(
            self.feature_extractor.get_features_dir(DatasetType.Train),
            self.feature_extractor.get_labels_filepath(DatasetType.Train),
        )
        validation_dataset = FeatureDataset(
            self.feature_extractor.get_features_dir(DatasetType.Validation),
            self.feature_extractor.get_labels_filepath(DatasetType.Validation),
        )
        test_dataset = FeatureDataset(
            self.feature_extractor.get_features_dir(DatasetType.Test),
            self.feature_extractor.get_labels_filepath(DatasetType.Test),
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
