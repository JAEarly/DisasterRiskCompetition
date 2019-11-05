"""LDA solution."""

import pickle

import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import models
from features import AlexNet256
from models import FeatureTrainer
from models import Model


class LDAModel(Model):
    """LDA model implementation."""

    def __init__(self, name, model_path=None):
        super().__init__(name)
        self.lda = None
        self.scaler = StandardScaler()
        if model_path is not None:
            self.load(model_path)

    def predict(self, feature_vector: torch.Tensor):
        feature_vector = self.scaler.transform(feature_vector.cpu().detach())
        clz = self.lda.predict(feature_vector)[0]
        # One hot encode
        prediction = [0] * self.num_classes
        prediction[clz] = 1
        return torch.tensor(prediction).float()

    def predict_batch(self, feature_vector_batch: torch.Tensor):
        feature_vector_batch = self.scaler.transform(
            feature_vector_batch.cpu().detach()
        )
        clzs = self.lda.predict(feature_vector_batch)
        predictions = []
        # One hot encode
        for clz in clzs:
            prediction = [0] * self.num_classes
            prediction[clz] = 1
            predictions.append(prediction)
        return torch.tensor(predictions).float()

    def load(self, path):
        with open(path, "rb") as file:
            self.lda, self.scaler = pickle.load(file)

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump((self.lda, self.scaler), file)

    def fit(self, features: torch.Tensor, labels):
        features = self.scaler.fit_transform(features.cpu().detach())
        self.lda = LinearDiscriminantAnalysis(n_components=3)
        self.lda.fit(features, labels)


if __name__ == "__main__":
    print("Creating LDA model")
    lda_model = models.LDAModel("lda_alexnet256")
    print("Creating feature extractor")
    trainer = FeatureTrainer(AlexNet256())
    trainer.train(lda_model)
