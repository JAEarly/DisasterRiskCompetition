"""
Base feature extractor implementation.

Using pre-trained models to take feature vectors from images.
"""

import os
import pickle
from abc import ABC, abstractmethod

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

    def extract(self, dataset_type: DatasetType) -> None:
        """
        Extract the features from an image dataset if not found.
        :param dataset_type: Dataset to use (training, test etc.)
        :return: None.
        """
        features_dir = self.get_features_dir(dataset_type)
        # Only extract if missing featres
        if not os.path.exists(features_dir) or len(os.listdir(features_dir)) != len(
            self.image_datasets.get_dataset(dataset_type)
        ):
            create_dirs_if_not_found(features_dir)
            if dataset_type is DatasetType.Competition:
                self._run_unlabelled_extraction()
            else:
                self._run_labelled_extraction(dataset_type)

    def _run_labelled_extraction(self, dataset_type: DatasetType) -> None:
        """
        Run the extraction for labelled data.
        :param dataset_type: Dataset to use (training, test etc.)
        :return: None.
        """
        i = 0
        labels = []
        dataset = self.image_datasets.get_dataset(dataset_type)
        for image, image_label in tqdm(
            dataset, desc="Extracting features  - " + dataset_type.name
        ):
            # Extract tensor and save
            feature_tensor = self.extractor_model(image.unsqueeze(0))
            self._save_tensor(dataset_type, feature_tensor, i)
            labels.append(image_label)
            i += 1
        # Save labels file
        labels = torch.tensor(labels)
        labels_filepath = self.get_labels_filepath(dataset_type)
        with open(labels_filepath, "wb") as file:
            pickle.dump(labels, file)

    def _run_unlabelled_extraction(self) -> None:
        """
        Run the extraction for unlabelled data (i.e. competition dataset).
        :return: None.
        """
        i = 0
        for image in tqdm(
            self.competition_dataset, desc="Extracting features  - competition",
        ):
            feature_tensor = self.extractor_model(image.unsqueeze(0))
            self._save_tensor(DatasetType.Competition, feature_tensor, i)
            i += 1

    def _save_tensor(self, dataset_type, tensor, idx) -> None:
        """
        Save a tensor to a file.
        :param dataset_type: Dataset in use (train, test etc.). Determines filepath.
        :param tensor: The tensor to save.
        :param idx: The index of the tensor in the dataset. Determines filename.
        :return: None.
        """
        feature_dir = self.get_features_dir(dataset_type)
        path = os.path.join(feature_dir, str(idx) + ".pkl")
        with open(path, "wb") as file:
            pickle.dump(tensor, file)

    def get_features_dir(self, dataset_type: DatasetType) -> str:
        """
        Get the directory for a set of features.
        :param dataset_type: Dataset in use (train, test etc.).
        :return: None.
        """
        return os.path.join(self.save_dir, dataset_type.name.lower(), self.name)

    def get_labels_filepath(self, dataset_type: DatasetType) -> str:
        """
        Get the filepath for the labels file.
        :param dataset_type: Dataset in use (train, test etc.).
        :return: None.
        """
        return os.path.join(self.save_dir, dataset_type.name.lower() + "_labels.pkl")


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
