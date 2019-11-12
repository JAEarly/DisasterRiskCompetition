"""Trainer base classes."""

import os
from abc import ABC
from abc import abstractmethod

from features import FeatureDatasets, FeatureExtractor, DatasetType, BalanceMethod
from utils import create_timestamp_str


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


class FeatureTrainer(Trainer):
    """Base implementation for feature based trainers."""

    def __init__(
        self, feature_extractor: FeatureExtractor, balance_method=BalanceMethod.NoSample
    ):
        self.feature_dataset = FeatureDatasets(
            feature_extractor, balance_method=balance_method
        )

    def train(self, model) -> (float, float):
        print("Loading features")
        features, labels = self.feature_dataset.get_features_and_labels(
            DatasetType.Train
        )
        print("Fitting model")
        model.fit(features, labels)
        print("Saving model")
        save_path = os.path.join(
            self.save_dir, model.name + "_" + create_timestamp_str() + ".pkl"
        )
        model.save(save_path)
