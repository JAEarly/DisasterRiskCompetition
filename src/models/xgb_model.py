"""XGBoost solution."""

import pickle

import torch
import xgboost as xgb

import features
from models import FeatureTrainer
from models import Model


class XGBModel(Model):
    """Model that uses XGBoost."""

    def __init__(self, model_path=None):
        super().__init__("xgb")
        self.xbg_bst = xgb.Booster()
        if model_path is not None:
            self.load(model_path)

    def predict(self, feature_tensor):
        feature = feature_tensor.cpu().detach().numpy()
        d_feature = xgb.DMatrix(feature)
        pred = self.xbg_bst.predict(d_feature)
        return torch.tensor(pred).float()

    def predict_batch(self, feature_batch):
        feature_batch = feature_batch.cpu().detach().numpy()
        d_batch = xgb.DMatrix(feature_batch)
        pred = self.xbg_bst.predict(d_batch)
        return torch.tensor(pred).float()

    def fit(self, training_features: torch.Tensor, labels):
        param = {
            "max_depth": 8,
            "eta": 1,
            "objective": "multi:softprob",
            "nthread": 4,
            "eval_metric": "auc",
            "verbosity": 2,
            "num_class": self.num_classes,
        }

        training_features = training_features.cpu().detach().numpy()
        print("Creating DMatrix")
        d_train = xgb.DMatrix(training_features, label=labels)
        print("Training")

        num_round = 5
        self.xbg_bst = xgb.train(param, d_train, num_round)

    def load(self, path):
        with open(path, "rb") as file:
            self.xbg_bst = pickle.load(file)

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.xbg_bst, file)


if __name__ == "__main__":
    feature_extractor = features.ResNetSMOTE()
    trainer = FeatureTrainer(
        feature_extractor, balance_method=features.BalanceMethod.NoSample
    )
    model = XGBModel()
    trainer.train(model)
