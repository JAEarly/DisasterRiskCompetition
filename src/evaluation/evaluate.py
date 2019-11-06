"""Model evaluation script."""

import torch
from sklearn.metrics import log_loss, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import features
import models
from features import DatasetType, FeatureDatasets


def evaluate(model: models.Model, data_loader: DataLoader, apply_softmax=True) -> None:
    """
    Evaluate a model on a given dataset.
    :param model: Model to evaluate.
    :param data_loader: Data loader wrapper around test set.
    :param apply_softmax: Should softmax be applied to the predictions.
    :return: None.
    """
    # Get truth and predictions
    y_true = []
    y_pred = []
    for batch, labels in tqdm(data_loader):
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

    # Print accuracy and log loss
    print("Accuracy:", accuracy_score(y_true, y_pred_classes))
    print("Log loss:", log_loss(y_true, y_probabilities, labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    _feature_extractor = features.ResNet18t256()
    _features_datasets = FeatureDatasets(_feature_extractor)
    _model = models.LDAModel(
        "resnet18t256_lda",
        model_path="./models/resnet18t256_lda_2019-11-06_16:09:08.pkl",
    )
    evaluate(_model, _features_datasets.get_loader(DatasetType.Test))
