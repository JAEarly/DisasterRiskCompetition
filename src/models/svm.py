"""SVM solution."""

import pickle

import numpy as np
import torch
from sklearn.svm import SVC

import models
from features import AlexNet256
from models import FeatureTrainer
from models import Model


class SVMModel(Model):
    """SVM model implementation."""

    def __init__(self, name, model_path=None):
        super().__init__(name)
        self.svm = None
        if model_path is not None:
            self.load(model_path)

    def predict(self, feature_vector: torch.Tensor):
        clz = self.svm.predict(feature_vector.detach().numpy())
        # One hot encode
        prediction = [0] * self.num_classes
        prediction[clz] = 1
        return torch.tensor(prediction).float()

    def predict_batch(self, feature_vector_batch: torch.Tensor):
        clzs = self.svm.predict(feature_vector_batch.detach().numpy())
        predictions = []
        # One hot encode
        for clz in clzs:
            prediction = [0] * self.num_classes
            prediction[clz] = 1
            predictions.append(prediction)
        return torch.tensor(predictions).float()

    def load(self, path):
        with open(path, "rb") as file:
            self.svm = pickle.load(file)

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.svm, file)

    def fit(self, features: torch.Tensor, labels):
        features, labels = self.reduce_training_data(features, labels)
        self.svm = SVC(verbose=True, kernel="poly", degree=8)
        self.svm.fit(features.detach().numpy(), labels)

    @staticmethod
    def reduce_training_data(features: torch.Tensor, labels):
        """
        Reduce training data to smaller amount.
        :param features: Training features.
        :param labels: Training labels.
        :return: Reduced features and labels.
        """
        return SVMModel._reduce_random(features, labels)

    @staticmethod
    def _reduce_equal(features: torch.Tensor, labels):
        """
        Reduce such that each class makes up an equal proportion.
        Based on the size of the smallest class.
        :param features: Training features.
        :param labels: Training labels.
        :return: Reduced features and labels.
        """
        _, counts = np.unique(labels, return_counts=True)
        class_size = min(counts)

        reduced_features = []
        reduced_labels = []
        class_counts = [0] * 5
        for feature_vector, label in zip(features, labels):
            if class_counts[label] < class_size:
                reduced_features.append(feature_vector)
                reduced_labels.append(label)
                class_counts[label] += 1
        reduced_features = torch.stack(reduced_features)
        return reduced_features, reduced_labels

    @staticmethod
    def _reduce_random(features: torch.Tensor, labels, reduction=0.2):
        """
        Sample the features at random.
        :param features: Training features.
        :param labels: Training labels.
        :param reduction: Proportion to reduce to.
        :return: Reduced features and labels.
        """
        num_features = features.shape[0]
        idxs = np.random.randint(
            low=0, high=num_features, size=int(reduction * num_features)
        )
        reduced_features = features[idxs]
        reduced_labels = labels[idxs]
        return reduced_features, reduced_labels


if __name__ == "__main__":
    print("Creating LDA model")
    svm_model = models.SVMModel("svm_alexnet256_poly8")
    print("Creating feature extractor")
    trainer = FeatureTrainer(AlexNet256())
    trainer.train(svm_model)
