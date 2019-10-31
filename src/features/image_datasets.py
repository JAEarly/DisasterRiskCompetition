from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms
from enum import Enum


class DatasetType(Enum):
    Train = 1
    Validation = 2
    Test = 3


class ImageDatasets:

    train_dir = "./data/processed/train"
    validation_dir = "./data/processed/validation"
    test_dir = "./data/processed/test"
    batch_size = 8

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.validation_dataset = datasets.ImageFolder(self.validation_dir, transform=self.transform)
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.transform)

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

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
            raise IndexError("Could not find dataset for type " + dataset_type.name)
