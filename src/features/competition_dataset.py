"""
Contains the abstract and implementation classes for the competition data.

Different to Datasets as it works with unlabelled data (i.e. class labels not given).
"""

import os
import pickle

from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset

from features import DatasetType


class CompetitionDataset(Dataset):
    """Abstract base class for competition dataset."""

    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.data_loader = data.DataLoader(
            self, batch_size=self.batch_size, shuffle=False
        )


class CompetitionImageDataset(CompetitionDataset):
    """Competition dataset backed by images."""

    data_dir = "./data/processed/competition"

    def __init__(self, transform=None):
        super().__init__()
        self.filenames = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, self.filenames[index]))
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class CompetitionFeatureDataset(CompetitionDataset):
    """Competition dataset backed with extracted features."""

    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.extract(DatasetType.Competition)
        self.data_dir = self.feature_extractor.get_features_dir(DatasetType.Competition)
        self.filenames = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_dir, self.filenames[index])
        with open(filepath, "rb") as file:
            feature = pickle.load(file)[0]
        return feature
