"""XGBoost solution."""

import pickle

import torch
import xgboost as xgb

import features
from models import Model
from training import FeatureTrainer


class XGBModel(Model):
    """Model that uses XGBoost."""

    def __init__(self, model_path=None):
        super().__init__("xgb", False)
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

    def fit(
        self,
        training_features: torch.Tensor,
        labels,
        val_features: torch.Tensor = None,
        val_labels=None,
        eta=0.3,
        gamma=0,
        depth=6,
        c_weight=1,
        reg_lambda=1,
        num_rounds=5,
    ):
        param = {
            # Fixed parameters
            "objective": "multi:softprob",
            "nthread": 4,
            "eval_metric": "mlogloss",
            "num_class": self.num_classes,
            # Hyper parameters
            "eta": eta,
            "gamma": gamma,
            "max_depth": depth,
            "min_child_weight": c_weight,
            "lambda": reg_lambda,
        }

        training_features = training_features.cpu().detach().numpy()
        print("Creating DMatrix")
        d_train = xgb.DMatrix(training_features, label=labels)
        d_val = None
        if val_features is not None and val_labels is not None:
            print("Using validation dataset for eval")
            val_features = val_features.cpu().detach().numpy()
            d_val = xgb.DMatrix(val_features, label=val_labels)
        print("Training")
        if d_val is not None:
            self.xbg_bst = xgb.train(
                param,
                d_train,
                num_rounds,
                evals=[[d_val, "ValidationSet"]],
                verbose_eval=True,
            )
        else:
            self.xbg_bst = xgb.train(param, d_train, num_rounds)

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
