from abc import ABC, abstractmethod

from torchvision import transforms


class Model(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, image_tensor):
        pass

    @abstractmethod
    def predict_batch(self, batch):
        pass

    def get_transform(self):
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()])
        return transform
