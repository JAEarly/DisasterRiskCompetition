import pandas as pd

from .model import Model


class BaselineModel(Model):

    def __init__(self):
        self.class_dist = pd.read_csv('./data/raw/train_labels.csv').groupby(['verified']).mean()[-1:].values[0]

    def predict(self, image_tensor):
        return self.class_dist

    def predict_batch(self, dataset_tensor):
        return [self.class_dist] * len(dataset_tensor)
