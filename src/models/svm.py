"""SVM solution."""

import pickle

import numpy as np
import torch
from sklearn.svm import SVC

from models import Model


class SVMModel(Model):
    """SVM model implementation."""

    def __init__(self, model_path=None):
        super().__init__("svm", False)
        self.svm = None
        if model_path is not None:
            self.load(model_path)

    def predict(self, feature_vector: torch.Tensor):
        prediction = self.svm.predict_proba(feature_vector.detach().cpu().numpy())
        # One hot encode
        # prediction = [0] * self.num_classes
        # prediction[clz] = 1
        return torch.tensor(prediction).float()

    def predict_batch(self, feature_vector_batch: torch.Tensor):
        prediction = self.svm.predict_proba(feature_vector_batch.detach().cpu().numpy())
        # predictions = []
        # One hot encode
        # for clz in clzs:
        #     prediction = [0] * self.num_classes
        #     prediction[clz] = 1
        #     predictions.append(prediction)
        return torch.tensor(prediction).float()

    def load(self, path):
        with open(path, "rb") as file:
            self.svm = pickle.load(file)

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.svm, file)

    def fit(self, features: torch.Tensor, labels, c=1.0, gamma='scale'):
        self.svm = SVC(C=c, gamma=gamma, probability=True)
        self.svm.fit(features.detach().cpu().numpy(), labels)
