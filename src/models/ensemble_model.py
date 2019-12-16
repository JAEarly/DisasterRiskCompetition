import torch
import os
from models import Model


class EnsembleModel(Model):

    def __init__(self, models, tag, apply_softmax, load=False):
        super().__init__("ensemble_" + tag, apply_softmax)
        self.models = models
        if load:
            self.load(self.get_models_dir())

    def predict(self, feature_tensor):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(feature_tensor))
        prediction = torch.stack(predictions).mean(dim=1)
        return prediction

    def predict_batch(self, feature_batch):
        predictions = []
        for model in self.models:
            predictions.append(model.predict_batch(feature_batch))
        predictions = torch.stack(predictions)
        prediction = predictions.mean(dim=0)
        return prediction

    def load(self, path):
        for i, model in enumerate(self.models):
            model_path = os.path.join(path, str(i) + ".pth")
            model.load(model_path)

    def save(self, path):
        for i, model in enumerate(self.models):
            model_path = os.path.join(path, str(i) + ".pth")
            model.save(model_path)

    def get_models_dir(self):
        return "./models/ensemble/" + self.name
