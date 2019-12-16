import torch
import os
from models import Model


class EnsembleModel(Model):

    def __init__(self, models, tag, apply_softmax):
        super().__init__("ensemble_" + tag, apply_softmax)
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

    def load(self, num_models):
        raise NotImplementedError()

    def save(self, path):
        for i, model in enumerate(self.models):
            model_path = os.path.join(path, str(i) + ".pth")
            model.save(model_path)

    def get_models_dir(self):
        return "./models/ensemble/" + self.name
