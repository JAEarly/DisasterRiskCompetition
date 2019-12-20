import torch
import os
from models import Model


class EnsembleModel(Model):

    def __init__(self, models, tag, load=False):
        super().__init__("ensemble_" + tag)
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


class PretrainedEnsembleModel(EnsembleModel):

    def predict(self, feature_tensor):
        predictions = []
        for model in self.models:
            model.net = model.net.to(model.device)
            predictions.append(model.predict(feature_tensor))
            model.net = model.net.to("cpu")
        prediction = torch.stack(predictions).mean(dim=1)
        return prediction

    def predict_batch(self, feature_batch):
        predictions = []
        for model in self.models:
            model.net = model.net.to(model.device)
            predictions.append(model.predict_batch(feature_batch))
            model.net = model.net.to("cpu")
        predictions = torch.stack(predictions)
        prediction = predictions.mean(dim=0)
        return prediction

