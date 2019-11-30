"""
Contains the abstract and implementation classes for the competition data.

Different to Datasets as it works with unlabelled data (i.e. class labels not given).
"""

import pickle

import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from features import DatasetType


class CompetitionLoader(DataLoader):
    def __init__(self, competition_dataset, batch_size=8):
        super().__init__(competition_dataset, batch_size=batch_size, shuffle=False)


class CompetitionImageDataset(ImageFolder):
    """Competition dataset backed by images."""

    data_dir = "./data/processed/competition"

    def __init__(self, transform):
        super().__init__(self.data_dir, transform=transform)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        file_name = os.path.basename(path)
        file_id = file_name[: file_name.index(".png")]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return file_id, sample


class CompetitionFeatureDataset(Dataset):
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
        return self.filenames[index].split(".")[0], feature
