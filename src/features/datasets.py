"""
Contains the abstract and implemented classes for a datasets wrapper.

A datasets class contains train, validation and test datasets (all labelled data).
These datasets are either backed by images (raw data) or features (extracted from images).
"""

import csv
import pickle
from enum import Enum
from typing import Tuple

import os
import torch
from abc import ABC, abstractmethod
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as torch_datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


CUSTOM_BALANCE = [0.083, 0.520, 0.037, 0.350, 0.010]


class DatasetType(Enum):
    """Enum for dataset types."""

    Train = 1
    Validation = 2
    Test = 3
    Competition = 4
    Pseudo = 5


class BalanceMethod(Enum):
    """Enum for dataset balancing methods."""

    NoSample = 0
    UnderSample = 1
    AvgSample = 2
    OverSample = 3
    CustomSample = 4


class Datasets(ABC):
    """Abstract base class for datasets."""

    def __init__(self, batch_size=8, balance_method=BalanceMethod.NoSample):
        self.batch_size = batch_size

        # Create datasets
        (
            self.train_dataset,
            self.validation_dataset,
            self.test_dataset,
            self.competition_dataset,
            self.pseudo_dataset,
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
        self.competition_loader = data.DataLoader(
            self.competition_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.pseudo_loader = data.DataLoader(
            self.pseudo_dataset, batch_size=self.batch_size, shuffle=True
        )

    @abstractmethod
    def create_datasets(
        self, balance_method: BalanceMethod
    ) -> Tuple[data.Dataset, data.Dataset, data.Dataset, data.Dataset, data.Dataset]:
        """
        Create the train, validation and test datasets.
        :param balance_method: Method for balancing the training dataset.
        :return: Train, validation, test and competition datasets as a tuple.
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
        if dataset_type == DatasetType.Competition:
            return self.competition_dataset
        if dataset_type == DatasetType.Pseudo:
            return self.pseudo_dataset
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
        if dataset_type == DatasetType.Competition:
            return self.competition_loader
        if dataset_type == DatasetType.Pseudo:
            return self.pseudo_loader
        raise IndexError("Could not find data loader for type " + dataset_type.name)


class ImageFolderWithFilenames(torch_datasets.ImageFolder):
    def __init__(self, image_dir, transform):
        super().__init__(image_dir, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def getitem_filename(self, index):
        path, _ = self.samples[index]
        sample, target = self[index]
        filename = os.path.basename(path).split(".")[0]
        return sample, target, filename


class ImageDatasets(Datasets):
    """Implementation of Datasets backed with images."""

    def __init__(self, transform: transforms.Compose = None, root_dir="./data/processed/"):
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
        self.train_dir = root_dir + "train"
        self.validation_dir = root_dir + "validation"
        self.test_dir = root_dir + "test"
        super().__init__()

    def create_datasets(self, balance_method: BalanceMethod):
        # Create image datasets using folder paths and given transform.
        train_dataset = ImageFolderWithFilenames(
            self.train_dir, transform=self.transform
        )
        validation_dataset = ImageFolderWithFilenames(
            self.validation_dir, transform=self.transform
        )
        test_dataset = ImageFolderWithFilenames(self.test_dir, transform=self.transform)
        competition_dataset = UnlabelledImageDataset("./data/processed/competition", transform=self.transform)
        pseudo_dataset = UnlabelledImageDataset("./data/processed/pseudo", transform=self.transform)

        return train_dataset, validation_dataset, test_dataset, competition_dataset, pseudo_dataset


class FeatureDataset(Dataset):
    """Feature dataset implementation. Uses pickled tensors for each feature vector."""

    def __init__(
        self, features_dir, labels_path, balance_method=BalanceMethod.NoSample
    ):
        super().__init__()
        self.data_dir = features_dir
        self.filenames = os.listdir(self.data_dir)
        self.path2label = self._load_labels(labels_path)
        if balance_method == BalanceMethod.NoSample:
            pass
        elif balance_method == BalanceMethod.UnderSample:
            self._undersample()
        elif balance_method == BalanceMethod.AvgSample:
            self._avgsample()
        elif balance_method == BalanceMethod.OverSample:
            self._oversample()
        elif balance_method == BalanceMethod.CustomSample:
            self._custom_sample()
        else:
            raise NotImplementedError("No balance method implemented for " + balance_method.name)

    def _undersample(self):
        data_dist = [0] * 5
        for label in self.path2label.values():
            data_dist[label] += 1
        min_class_size = min(data_dist)
        self._sample_to_target_dist([min_class_size] * 5)

    def _avgsample(self):
        data_dist = [0] * 5
        for label in self.path2label.values():
            data_dist[label] += 1
        avg_class_size = int(sum(data_dist) / 5)
        self._sample_to_target_dist([avg_class_size] * 5)

    def _oversample(self):
        data_dist = [0] * 5
        for label in self.path2label.values():
            data_dist[label] += 1
        max_class_size = max(data_dist)
        self._sample_to_target_dist([max_class_size] * 5)

    def _custom_sample(self):
        data_dist = [0] * 5
        for label in self.path2label.values():
            data_dist[label] += 1
        max_class_size = max(data_dist)
        target_dist = [0] * 5
        for i in range(5):
            target_dist[i] = int(CUSTOM_BALANCE[i] * max_class_size)
        self._sample_to_target_dist(target_dist)

    def _sample_to_target_dist(self, target_dist):
        data_dist = [0] * 5
        reduced_filenames = []
        reduced_labels = []

        for i in range(5):
            while data_dist[i] < target_dist[i]:
                for filename in self.filenames:
                    label = self.path2label[filename]
                    if label == i:
                        reduced_filenames.append(filename)
                        reduced_labels.append(label)
                        data_dist[label] += 1
                        if data_dist[i] == target_dist[i]:
                            break

        self.filenames = reduced_filenames
        self.labels = reduced_labels

    def _load_labels(self, filepath):
        path2label = {}
        with open(filepath, mode="r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                path2label[row[0] + ".pkl"] = int(row[1])
        return path2label

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
        filename = self.filenames[index]
        return self._load_vector(filename), self.path2label[filename]

    def getitem_filename(self, index):
        filename = self.filenames[index]
        file_id = os.path.basename(filename).split(".")[0]
        return self._load_vector(filename), self.path2label[filename], file_id


class FeatureDatasets(Datasets):
    """Implementation of Datasets back with a feature extractor."""

    def __init__(self, feature_extractor, balance_method=BalanceMethod.NoSample):
        # Ensure features are extracted
        self.feature_extractor = feature_extractor
        self.feature_extractor.extract(DatasetType.Train)
        self.feature_extractor.extract(DatasetType.Validation)
        self.feature_extractor.extract(DatasetType.Test)
        self.feature_extractor.extract(DatasetType.Competition)
        self.feature_extractor.extract(DatasetType.Pseudo)
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
        competition_dataset = UnlabelledFeatureDataset(
            self.feature_extractor, DatasetType.Competition
        )
        pseudo_dataset = UnlabelledFeatureDataset(
            self.feature_extractor, DatasetType.Pseudo
        )
        return train_dataset, validation_dataset, test_dataset, competition_dataset, pseudo_dataset

    def get_features_and_labels(
        self, dataset_type: DatasetType
    ) -> (torch.Tensor, torch.Tensor):
        """
        Get all the features and labels for a dataset.
        :param dataset_type: Dataset to fetch labels for.
        :return: Tensor of features and tensor of labels.
        """
        if dataset_type == DatasetType.Competition:
            raise ValueError("Cannot get features and labels for unlabelled dataset competition data.")
        return self.get_features_and_labels_from_dataloader(self.get_loader(dataset_type))

    def get_features_and_labels_from_dataloader(self, data_loader: DataLoader):
        features = []
        labels = []
        for batch, batch_labels in data_loader:
            features.extend(batch)
            labels.extend(batch_labels)
        features = torch.stack(features)
        labels = torch.tensor(labels)
        return features, labels

    def get_features(
        self, dataset_type: DatasetType
    ) -> (torch.Tensor, torch.Tensor):
        features = []
        for batch, _ in self.get_loader(dataset_type):
            features.extend(batch)
        features = torch.stack(features)
        return features


class UnlabelledImageDataset(ImageFolder):
    """Competition dataset backed by images."""

    def __init__(self, data_dir, transform=None):
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        super().__init__(data_dir, transform=transform)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        file_name = os.path.basename(path)
        file_id = file_name[: file_name.index(".png")]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, file_id


class UnlabelledFeatureDataset(Dataset):
    """Competition dataset backed with extracted features."""

    def __init__(self, feature_extractor, dataset_type: DatasetType):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.extract(DatasetType.Competition)
        self.data_dir = self.feature_extractor.get_features_dir(dataset_type)
        self.filenames = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_dir, self.filenames[index])
        with open(filepath, "rb") as file:
            feature = pickle.load(file)[0]
        return feature, self.filenames[index].split(".")[0]
