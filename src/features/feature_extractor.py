import os
import pickle
from abc import ABC, abstractmethod

import torch
from torch import nn
from tqdm import tqdm

from features import ImageDatasets, DatasetType, CompetitionImageDataset
from utils import create_dirs_if_not_found


class FeatureExtractor(ABC):

    save_dir = "./models/features/"

    def __init__(self, name):
        self.name = name
        self.extractor_model = self.setup_model()
        self.image_datasets = ImageDatasets(self.get_transform())
        self.competition_dataset = CompetitionImageDataset(self.get_transform())

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def get_transform(self):
        pass

    def extract(self, dataset_type: DatasetType) -> (torch.Tensor, torch.Tensor):
        features, labels = self.load(dataset_type)
        if features is None or (labels is None and dataset_type is not DatasetType.Competition):
            if dataset_type is DatasetType.Competition:
                features = self._run_unlabelled_extraction()
            else:
                features, labels = self._run_labelled_extraction(dataset_type)
            self.save(features, labels, dataset_type)
        if labels is not None:
            labels = torch.tensor(labels)
        return features, labels

    def _run_labelled_extraction(self, dataset_type: DatasetType):
        features = []
        labels = []
        dataset_loader = self.image_datasets.get_loader(dataset_type)
        for batch, batch_labels in tqdm(dataset_loader, desc='Extracting features  - ' + dataset_type.name):
            features.extend(self.extractor_model(batch))
            labels.extend(batch_labels.cpu().detach().numpy())
        features = torch.stack(features)
        features = features.cpu().detach()
        return features, labels

    def _run_unlabelled_extraction(self):
        features = []
        for batch in tqdm(self.competition_dataset.data_loader, desc='Extracting features  - competition'):
            features.extend(self.extractor_model(batch))
        features = torch.stack(features)
        features = features.cpu().detach()
        return features

    def save(self, features, labels, dataset_type: DatasetType):
        create_dirs_if_not_found(self.save_dir)
        features_filepath = self._get_features_filepath(dataset_type)
        labels_filepath = self._get_labels_filepath(dataset_type)
        with open(features_filepath, "wb") as file:
            pickle.dump(features, file)
        if labels is not None:
            with open(labels_filepath, "wb") as file:
                pickle.dump(labels, file)

    def load(self, dataset_type: DatasetType):
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

    def _get_features_filepath(self, dataset_type: DatasetType):
        return os.path.join(self.save_dir, self.name + "_" + dataset_type.name.lower() + "_features.pkl")

    def _get_labels_filepath(self, dataset_type: DatasetType):
        return os.path.join(self.save_dir, self.name + "_" + dataset_type.name.lower() + "_labels.pkl")


class IdentityLayer(nn.Module):
    """
    Identity layer that just passes input through to output.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
