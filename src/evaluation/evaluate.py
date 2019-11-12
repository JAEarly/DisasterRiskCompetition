"""Model evaluation script."""

import time

import torch
from sklearn.metrics import log_loss, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import features
import models
from features import DatasetType, FeatureDatasets


def evaluate(
    model: models.Model, data_loader: DataLoader, apply_softmax=True
) -> (float, float):
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

    # Print accuracy and log loss
    acc = accuracy_score(y_true, y_pred_classes)
    ll = log_loss(y_true, y_probabilities, labels=[0, 1, 2, 3, 4])
    print("Accuracy: {:.3f}".format(acc))
    print("Log loss: {:.3f}".format(ll))

    return acc, ll


if __name__ == "__main__":
    _feature_extractor = features.AlexNet()
    _features_datasets = FeatureDatasets(_feature_extractor)
    _model = models.NNModel(
        models.LinearNN,
        _feature_extractor.feature_size,
        state_dict_path="./models/grid_search_alexnet_linearnn/alexnet_linearnn_best.pth",
        eval_mode=True,
    )

    print("Training Set Results")
    train_acc, train_loss = evaluate(
        _model, _features_datasets.get_loader(DatasetType.Train)
    )

    time.sleep(0.1)
    print("")
    print("Validation Set Results")
    val_acc, val_loss = evaluate(
        _model, _features_datasets.get_loader(DatasetType.Validation)
    )

    time.sleep(0.1)
    print("")
    print("Test Set Results")
    test_acc, test_loss = evaluate(
        _model, _features_datasets.get_loader(DatasetType.Test)
    )

    time.sleep(0.1)
    print("")
    print("Output for results.md")
    print(
        ("{:.3f} | " * 6).format(
            train_acc, train_loss, val_acc, val_loss, test_acc, test_loss
        )
    )
