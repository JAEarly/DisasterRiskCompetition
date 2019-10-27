import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CompetitionDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
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
