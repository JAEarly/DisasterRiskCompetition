"""Base model implementation."""

from abc import ABC, abstractmethod

import torch


class Model(ABC):
    """Base class for models."""

    def __init__(self, name, apply_softmax, num_classes=5):
        self.name = name
        self.apply_softmax = apply_softmax
        self.num_classes = num_classes

    @abstractmethod
    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Predict a label for an input.
        :param tensor: Tensor (image or feature)
        :return: One hot encoding of class.
        """

    @abstractmethod
    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Predict labels for a batch of inputs.
        :param batch: Tensor of n images (or features).
        :return: One hot encoding of the predicted class for each input.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model.
        :param path: Path to model file.
        :return: None.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model.
        :param path: Path of save location.
        :return: None.
        """
