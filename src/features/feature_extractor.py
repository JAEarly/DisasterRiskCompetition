"""
Base feature extractor implementation.

Using pre-trained models to take feature vectors from images.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from features import ImageDatasets, DatasetType, CompetitionImageDataset
from utils import create_dirs_if_not_found


class FeatureExtractor(ABC):
    """Base class for feature extractor."""

    save_dir = "./models/features/"

    def __init__(self, name):
        self.name = name
        self.extractor_model = self.setup_model()
        self.image_datasets = ImageDatasets(self.get_transform())
        self.competition_dataset = CompetitionImageDataset(self.get_transform())

    @abstractmethod
    def setup_model(self) -> nn.Module:
        """
        Create the extraction model.
        :return: The pre-trained model with the correct output size.
        """

    @abstractmethod
    def get_transform(self) -> transforms.Compose:
        """
        Get the image transform used by the feature extractor.
        :return: The torchvision image transform.
        """

    def extract(self, dataset_type: DatasetType) -> (torch.Tensor, torch.Tensor):
        """
        Extract a dataset into features and labels.

        Will attempt to find existing extracted information, if not will run extraction.
        :param dataset_type: Which dataset to extract (training, validation or test)
        :return: The extracted features and their labels.
        """
        # Attempt to load existing dataset
        features, labels = self.load(dataset_type)
        # If missing information, run extraction
        if features is None or (
            labels is None and dataset_type is not DatasetType.Competition
        ):
            if dataset_type is DatasetType.Competition:
                features = self._run_unlabelled_extraction()
            else:
                features, labels = self._run_labelled_extraction(dataset_type)
            self.save(features, labels, dataset_type)
        return features, labels

    def _run_labelled_extraction(
        self, dataset_type: DatasetType
    ) -> (torch.Tensor, torch.Tensor):
        """
        Run a labelled extraction from a dataset.
        :param dataset_type: Which dataset to extract (training, validation or test)
        :return: A
        """
        features = []
        labels = []
        dataset_loader = self.image_datasets.get_loader(dataset_type)
        for batch, batch_labels in tqdm(
            dataset_loader, desc="Extracting features  - " + dataset_type.name
        ):
            features.extend(self.extractor_model(batch))
            labels.extend(batch_labels.cpu().detach().numpy())
        features = torch.stack(features).cpu().detach()
        labels = torch.tensor(labels)
        return features, labels

    def _run_unlabelled_extraction(self) -> torch.Tensor:
        """
        Run an unlabelled extraction over the competition dataset (i.e. no class labels).
        :return: The tensor of extracted features.
        """
        features = []
        for batch in tqdm(
            self.competition_dataset.data_loader,
            desc="Extracting features  - competition",
        ):
            features.extend(self.extractor_model(batch))
        features = torch.stack(features)
        features = features.cpu().detach()
        return features

    def save(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor],
        dataset_type: DatasetType,
    ) -> None:
        """
        Save a set of features and labels.
        :param features: Features to save.
        :param labels: Labels to save (can be None if saving an unlabelled dataset).
        :param dataset_type: Dataset type being saved (determines save path).
        :return: None.
        """
        create_dirs_if_not_found(self.save_dir)
        features_filepath = self._get_features_filepath(dataset_type)
        labels_filepath = self._get_labels_filepath(dataset_type)
        with open(features_filepath, "wb") as file:
            pickle.dump(features, file)
        if labels is not None:
            with open(labels_filepath, "wb") as file:
                pickle.dump(labels, file)

    def load(
        self, dataset_type: DatasetType
    ) -> (Optional[torch.Tensor], Optional[torch.Tensor]):
        """
        Attempt to load features and labels for a dataset.
        :param dataset_type: Dataset to load for (determines file paths).
        :return: Features and labels if found.
        """
        features_filepath = self._get_features_filepath(dataset_type)
        labels_filepath = self._get_labels_filepath(dataset_type)
        features = None
        labels = None
        if os.path.exists(features_filepath):
            with open(features_filepath, "rb") as file:
                features = pickle.load(file)
        if os.path.exists(labels_filepath):
            with open(labels_filepath, "rb") as file:
                labels = pickle.load(file)
        return features, labels

    def _get_features_filepath(self, dataset_type: DatasetType) -> str:
        """
        Get the features filepath based on the dataset type.
        :param dataset_type: Which dataset to generate the path for.
        :return: Filepath as string.
        """
        return os.path.join(
            self.save_dir, self.name + "_" + dataset_type.name.lower() + "_features.pkl"
        )

    def _get_labels_filepath(self, dataset_type: DatasetType) -> str:
        """
        Get the labels filepath based on the dataset type.
        :param dataset_type: Which dataset to generate the path for.
        :return: Filepath as string.
        """
        return os.path.join(
            self.save_dir, self.name + "_" + dataset_type.name.lower() + "_labels.pkl"
        )


class IdentityLayer(nn.Module):
    """
    Identity layer that just passes input through to output.

    Used for replacing the final layer of networks for transfer learning.
    """

    def forward(self, x):
        """
        Pass input straight through this layer.
        :param x: Input to network layer.
        :return: Output of layer (same as input).
        """
        return x
