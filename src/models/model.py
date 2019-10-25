from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def predict(self, image_tensor):
        pass

    @abstractmethod
    def get_transform(self):
        pass
