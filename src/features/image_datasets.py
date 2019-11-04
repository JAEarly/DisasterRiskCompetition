from torchvision import datasets
from torchvision.transforms import transforms

from features import Datasets


class ImageDatasets(Datasets):

    train_dir = "./data/processed/train"
    validation_dir = "./data/processed/validation"
    test_dir = "./data/processed/test"

    def __init__(self, transform: transforms.Compose):
        self.transform = transform
        super().__init__()

    def create_datasets(self):
        train_dataset = datasets.ImageFolder(self.train_dir, transform=self.transform)
        validation_dataset = datasets.ImageFolder(self.validation_dir, transform=self.transform)
        test_dataset = datasets.ImageFolder(self.test_dir, transform=self.transform)

        return train_dataset, validation_dataset, test_dataset
