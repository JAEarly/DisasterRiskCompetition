from abc import ABC, abstractmethod

from torch.utils import data
from torchvision import datasets

from models import Model


class Trainer(ABC):

    data_dir = "./data/processed/train"
    save_dir = "./models"
    num_classes = 5
    batch_size = 8
    test_proportion = 0.2

    def __init__(self, model: Model):
        self.model = model

        self.full_dataset = datasets.ImageFolder(self.data_dir, transform=self.model.get_transform())

        self.train_size = int((1 - self.test_proportion) * len(self.full_dataset))
        self.test_size = len(self.full_dataset) - self.train_size

        self.train_dataset, self.test_dataset = data.random_split(self.full_dataset, [self.train_size, self.test_size])
        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    @abstractmethod
    def train(self):
        pass
