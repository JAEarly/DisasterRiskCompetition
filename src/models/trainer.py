from abc import ABC, abstractmethod


class Trainer(ABC):

    save_dir = "./models"
    num_classes = 5

    #def __init__(self):
        # TODO extract into image trainer
        #self.image_datasets = ImageDatasets(model.get_transform())

    @abstractmethod
    def train(self, model, class_weight=None):
        pass
