from abc import ABC, abstractmethod
from enum import Enum

from torch.utils import data


class DatasetType(Enum):
    Train = 1
    Validation = 2
    Test = 3


class Datasets(ABC):

    def __init__(self, batch_size=8):
        self.batch_size = batch_size

        self.train_dataset, self.validation_dataset, self.test_dataset = self.create_datasets()

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    @abstractmethod
    def create_datasets(self):
        pass

    def get_dataset(self, dataset_type: DatasetType):
        if dataset_type == DatasetType.Train:
            return self.train_dataset
        elif dataset_type == DatasetType.Validation:
            return self.validation_dataset
        elif dataset_type == DatasetType.Test:
            return self.test_dataset
        else:
            raise IndexError("Could not find dataset for type " + dataset_type.name)

    def get_loader(self, dataset_type: DatasetType):
        if dataset_type == DatasetType.Train:
            return self.train_loader
        elif dataset_type == DatasetType.Validation:
            return self.validation_loader
        elif dataset_type == DatasetType.Test:
            return self.test_loader
        else:
            raise IndexError("Could not find data loader for type " + dataset_type.name)
