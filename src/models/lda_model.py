import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import Tensor

from .model import Model
from sklearn.preprocessing import StandardScaler


class LDAModel(Model):

    def __init__(self, name, model_path=None):
        super().__init__(name)
        self.lda = None
        self.scaler = StandardScaler()
        if model_path is not None:
            self.load(model_path)

    def predict(self, feature_vector: Tensor):
        feature_vector = self.scaler.transform(feature_vector.cpu().detach())
        clz = self.lda.predict(feature_vector)[0]
        prediction = [0] * self.num_classes
        prediction[clz] = 1
        return prediction

    def predict_batch(self, feature_vector_batch: Tensor):
        feature_vector_batch = self.scaler.transform(feature_vector_batch.cpu().detach())
        clzs = self.lda.predict(feature_vector_batch)
        predictions = []
        for clz in clzs:
            prediction = [0] * self.num_classes
            prediction[clz] = 1
            predictions.append(prediction)
        return predictions

    def load(self, path):
        with open(path, "rb") as file:
            self.lda, self.scaler = pickle.load(file)

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump((self.lda, self.scaler), file)

    def fit(self, features: Tensor, labels):
        features = self.scaler.fit_transform(features.cpu().detach())
        self.lda = LinearDiscriminantAnalysis(n_components=3)
        self.lda.fit(features, labels)

