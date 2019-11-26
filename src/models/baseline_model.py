"""Baseline model that predicts based on training class distribution."""

import pandas as pd
import torch

from .model import Model


class BaselineModel(Model):
    """Baseline model implementation."""

    def __init__(self):
        super().__init__("baseline", False)
        self.class_dist = (
            pd.read_csv("./data/raw/train_labels.csv")
            .groupby(["verified"])
            .mean()[-1:]
            .values[0]
        )

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.class_dist)

    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.class_dist] * len(batch))

    def load(self, path: str) -> None:
        pass

    def save(self, path: str) -> None:
        pass
