import pickle

from sklearn.cluster import KMeans
from torch import Tensor

from .model import Model


class KMeansModel(Model):

    def __init__(self, name, model_path=None):
        super().__init__(name)
        self.km = None
        if model_path is not None:
            self.load(model_path)

    def predict(self, feature_vector: Tensor):
        clz = self.km.predict(feature_vector.cpu().detach())[0]
        prediction = [0] * self.num_classes
        prediction[clz] = 1
        return prediction

    def predict_batch(self, feature_vector_batch: Tensor):
        clzs = self.km.predict(feature_vector_batch.cpu().detach())
        predictions = []
        for clz in clzs:
            prediction = [0] * self.num_classes
            prediction[clz] = 1
            predictions.append(prediction)
        return predictions

    def load(self, path: str):
        with open(path, "rb") as file:
            self.km = pickle.load(file)

    def save(self, path: str):
        with open(path, "wb") as file:
            pickle.dump(self.km, file)

    def fit(self, feature_vectors: Tensor):
        self.km = KMeans(
            n_clusters=self.num_classes, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0,
            verbose=1
        )
        self.km.fit(feature_vectors.cpu().detach())
