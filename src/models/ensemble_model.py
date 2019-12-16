import torch

from models import Model


class EnsembleModel(Model):
    def __init__(self, models, tag):
        super().__init__("ensemble_" + tag, True)
        self.models = models

    def predict(self, feature_tensor):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(feature_tensor))
        prediction = torch.stack(predictions).mean(dim=1)
        return prediction

    def predict_batch(self, feature_batch):
        predictions = []
        # print(len(self.models))
        for model in self.models:
            predictions.append(model.predict_batch(feature_batch))
        # print(predictions[0].shape)
        predictions = torch.stack(predictions)
        # print(predictions.shape)
        prediction = predictions.mean(dim=0)
        # print(prediction.shape)
        # exit(0)
        return prediction

    def load(self, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()
