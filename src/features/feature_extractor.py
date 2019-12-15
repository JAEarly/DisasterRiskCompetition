"""
Base feature extractor implementation.

Using pre-trained models to take feature vectors from images.
"""

import csv
import pickle

import os
import torch
from abc import ABC, abstractmethod
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from features import ImageDatasets, DatasetType
from utils import create_dirs_if_not_found


class FeatureExtractor(ABC):
    """Base class for feature extractor."""

    def __init__(self, name, save_dir="./models/features/", root_dir="./data/processed/"):
        self.name = name
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.extractor_model, self.feature_size = self.setup_model()
        self.image_datasets = ImageDatasets(self.get_transform(), root_dir=self.root_dir)

    @abstractmethod
    def setup_model(self) -> (nn.Module, int):
        """
        Create the extraction model.
        :return: The pre-trained model with the correct output size.
        """

    def get_transform(self) -> transforms.Compose:
        """
        Create a transform that reduces to images to 256 x 256.
        :return: Composed transform.
        """
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
        return transform

    def extract(self, dataset_type: DatasetType) -> None:
        """
        Extract the features from an image dataset if not found.
        :param dataset_type: Dataset to use (training, test etc.)
        :return: None.
        """
        features_dir = self.get_features_dir(dataset_type)
        image_dataset = self.image_datasets.get_dataset(dataset_type)
        # Only extract if missing features
        if not os.path.exists(features_dir) or len(os.listdir(features_dir)) != len(
            image_dataset
        ):
            # CUDA setup
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print("Running feature extraction using", device)

            create_dirs_if_not_found(features_dir)
            if dataset_type in [DatasetType.Competition, DatasetType.Pseudo]:
                self._run_unlabelled_extraction(dataset_type, device)
            else:
                self._run_labelled_extraction(dataset_type, device)

    def _run_labelled_extraction(self, dataset_type: DatasetType, device: str) -> None:
        """
        Run the extraction for labelled data.
        :param dataset_type: Dataset to use (training, test etc.)
        :return: None.
        """
        dataset = self.image_datasets.get_dataset(dataset_type)
        self.extractor_model = self.extractor_model.to(device)

        filenames = []
        labels = []
        for i in tqdm(
            range(len(dataset)), desc="Extracting features  - " + dataset_type.name
        ):
            image, image_label, filename = dataset.getitem_filename(i)
            # Extract tensor and save
            feature_tensor = self.extractor_model(image.unsqueeze(0).to(device))
            self._save_tensor(dataset_type, feature_tensor, filename)
            filenames.append(filename)
            labels.append(image_label)

        # Save labels file
        labels_filepath = self.get_labels_filepath(dataset_type)
        with open(labels_filepath, "w+") as file:
            csv_writer = csv.writer(file)
            for filename, label in zip(filenames, labels):
                csv_writer.writerow([filename, label])

    def _run_unlabelled_extraction(self, dataset_type: DatasetType, device: str) -> None:
        """
        Run the extraction for unlabelled data (i.e. competition dataset).
        :return: None.
        """
        dataset = self.image_datasets.get_dataset(dataset_type)
        self.extractor_model = self.extractor_model.to(device)

        for image, file_id in tqdm(
            dataset, desc="Extracting features - " + dataset_type.name,
        ):
            feature_tensor = self.extractor_model(image.unsqueeze(0).to(device))
            self._save_tensor(dataset_type, feature_tensor, file_id)

    def _save_tensor(self, dataset_type, tensor, file_id) -> None:
        """
        Save a tensor to a file.
        :param dataset_type: Dataset in use (train, test etc.). Determines filepath.
        :param tensor: The tensor to save.
        :param file_id: The original filename of the image.
        :return: None.
        """
        feature_dir = self.get_features_dir(dataset_type)
        path = os.path.join(feature_dir, str(file_id) + ".pkl")
        with open(path, "wb") as file:
            pickle.dump(tensor, file)

    def get_features_dir(self, dataset_type: DatasetType) -> str:
        """
        Get the directory for a set of features.
        :param dataset_type: Dataset in use (train, test etc.).
        :return: None.
        """
        return os.path.join(self.save_dir, self.name, dataset_type.name.lower())

    def get_labels_filepath(self, dataset_type: DatasetType) -> str:
        """
        Get the filepath for the labels file.
        :param dataset_type: Dataset in use (train, test etc.).
        :return: None.
        """
        return os.path.join(
            self.save_dir, self.name, dataset_type.name.lower() + "_labels.csv"
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
