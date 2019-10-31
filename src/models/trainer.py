from abc import ABC, abstractmethod

from features import ImageDatasets
from models import Model


class Trainer(ABC):

    save_dir = "./models"
    num_classes = 5

    def __init__(self, model: Model):
        self.model = model
        self.image_datasets = ImageDatasets(model.get_transform())

    @abstractmethod
    def train(self, class_weight=None):
        pass
