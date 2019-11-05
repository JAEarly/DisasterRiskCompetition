"""Trainer base classes."""

from abc import ABC
from abc import abstractmethod

from features import FeatureDatasets, FeatureExtractor


class Trainer(ABC):
    """Base implementation for trainer classes."""

    save_dir = "./models"

    @abstractmethod
    def train(self, model, class_weight=None) -> None:
        """
        Train a model.
        :param model: Model to train.
        :param class_weight: Weight each class during training.
        :return: None.
        """


class FeatureTrainer(Trainer, ABC):
    """Base implementation for feature based trainers."""

    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_dataset = FeatureDatasets(feature_extractor)
