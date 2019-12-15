import pickle

import os
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class PseudoFeatureDataset(Dataset):

    def __init__(
        self, model, features_dir, threshold=0.95
    ):
        super().__init__()
        self.data_dir = features_dir
        self.filenames, self.path2label = self._threshold(model, threshold)

    def _threshold(self, model, threshold):
        all_filenames = os.listdir(self.data_dir)
        filenames = []
        path2label = {}
        for filename in tqdm(all_filenames, desc="Thresholding pseudo dataset", leave=False):
            feature = self._load_vector(filename)
            prediction = model.predict(feature).cpu().detach()
            if model.apply_softmax:
                prediction = torch.softmax(prediction, 0)
            confidence = max(prediction)
            if confidence > threshold:
                filenames.append(filename)
                path2label[filename] = prediction.argmax().item()
        print(str(len(filenames)) + "/" + str(len(all_filenames)), "datapoints above threshold of", threshold)
        return filenames, path2label

    def _load_vector(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "rb") as file:
            feature = pickle.load(file)
            if len(feature.shape) == 2:
                feature = feature[0]
        return feature

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        return self._load_vector(filename), self.path2label[filename]
