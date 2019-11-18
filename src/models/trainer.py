"""Trainer base classes."""

from enum import Enum

import os
import pandas as pd
import torch
from abc import ABC
from abc import abstractmethod
from sklearn.metrics import log_loss, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from features import FeatureDatasets, FeatureExtractor, DatasetType, BalanceMethod
from utils import create_timestamp_str


class ClassWeightMethod(Enum):
    """Enum for class weight methods."""

    Unweighted = 1
    SumBased = 2
    MaxBased = 3


class Trainer(ABC):
    """Base implementation for trainer classes."""

    save_dir = "./models"

    @abstractmethod
    def train(self, model) -> (float, float):
        """
        Train a model.
        :param model: Model to train.
        :return: None.
        """

    @staticmethod
    def evaluate(
            model, data_loader: DataLoader, apply_softmax=True, verbose=False
    ) -> (float, float):
        """
        Evaluate a model on a given dataset.
        :param model: Model to evaluate.
        :param data_loader: Data loader wrapper around test set.
        :param apply_softmax: Should softmax be applied to the predictions.
        :param verbose: Output results (true) or just return them (false).
        :return: None.
        """
        # Get truth and predictions
        y_true = []
        y_pred = []
        for batch, labels in tqdm(data_loader, leave=False):
            y_pred.extend(model.predict_batch(batch))
            y_true.extend(labels)

        # Format as tensors
        y_true = torch.stack(y_true)
        y_pred = torch.stack(y_pred).cpu().detach()

        # Convert from one hot to class ids
        _, y_pred_classes = y_pred.max(1)

        # Calculate prediction probabilities if required
        if apply_softmax:
            y_probabilities = torch.softmax(y_pred, 1)
        else:
            y_probabilities = y_pred

        y_true_pd = pd.Series(y_true, name="Actual")
        y_pred_pd = pd.Series(y_pred_classes, name="Predicted")
        conf_mat = pd.crosstab(
            y_true_pd, y_pred_pd, rownames=["Actual"], colnames=["Predicted"], margins=True
        )

        # Print accuracy and log loss
        acc = accuracy_score(y_true, y_pred_classes)
        ll = log_loss(y_true, y_probabilities, labels=[0, 1, 2, 3, 4])
        if verbose:
            print("Accuracy: {:.3f}".format(acc))
            print("Log loss: {:.3f}".format(ll))
            print("Confusion matrix")
            print(conf_mat)

        return acc, ll


class FeatureTrainer(Trainer):
    """Base implementation for feature based trainers."""

    def __init__(
        self, feature_extractor: FeatureExtractor, balance_method=BalanceMethod.NoSample
    ):
        self.feature_dataset = FeatureDatasets(
            feature_extractor, balance_method=balance_method
        )

    def train(self, model, **kwargs) -> (float, float):
        print("Loading features")
        features, labels = self.feature_dataset.get_features_and_labels(
            DatasetType.Train
        )
        print("Fitting model")
        model.fit(features, labels, **kwargs)
        print("Saving model")
        save_path = os.path.join(
            self.save_dir, model.name + "_" + create_timestamp_str() + ".pkl"
        )
        model.save(save_path)

        val_acc, val_loss = self.evaluate(
            model,
            self.feature_dataset.get_loader(DatasetType.Validation),
            apply_softmax=True,
        )
        return val_acc, val_loss
