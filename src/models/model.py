from abc import ABC, abstractmethod

from torchvision import transforms


class Model(ABC):

    num_classes = 5

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, image_tensor):
        pass

    @abstractmethod
    def predict_batch(self, batch):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()])
        return transform
