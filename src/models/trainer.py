from abc import ABC, abstractmethod

from torch.utils import data
from torchvision import datasets

from models import Model


class Trainer(ABC):

    train_dir = "./data/processed/train"
    validation_dir = "./data/processed/validation"
    test_dir = "./data/processed/test"
    save_dir = "./models"
    num_classes = 5
    batch_size = 8

    def __init__(self, model: Model):
        self.model = model

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.model.get_transform())
        self.validation_dataset = datasets.ImageFolder(self.validation_dir, transform=self.model.get_transform())
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.model.get_transform())

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    @abstractmethod
    def train(self):
        pass
