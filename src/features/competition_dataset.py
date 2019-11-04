import os

from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset

from features import DatasetType


class CompetitionDataset(Dataset):

    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.data_loader = data.DataLoader(self, batch_size=self.batch_size, shuffle=False)


class CompetitionImageDataset(CompetitionDataset):

    data_dir = "./data/processed/competition"

    def __init__(self, transform=None):
        super().__init__()
        self.id_list = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, self.id_list[index]))
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


class CompetitionFeatureDataset(CompetitionDataset):

    def __init__(self, feature_extractor):
        super().__init__()
        self.features, _ = feature_extractor.extract(DatasetType.Competition)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]
