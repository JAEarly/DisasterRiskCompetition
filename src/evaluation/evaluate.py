"""Model evaluation script."""

import torch
from sklearn.metrics import log_loss, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import features
import models
from features import DatasetType, FeatureDatasets


def evaluate(model: models.Model, data_loader: DataLoader) -> None:
    """
    Evaluate a model on a given dataset.
    :param model: Model to evaluate.
    :param data_loader: Data loader wrapper around test set.
    :return: None.
    """
    # Get truth and predictions
    y_true = []
    y_pred = []
    for batch, labels in tqdm(data_loader):
        y_pred.extend(model.predict(batch))
        y_true.extend(labels)

    # Format as tensors
    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred).cpu().detach()

    # Convert from one hot to class ids
    _, y_pred_classes = y_pred.max(1)

    # Calculate prediction probabilities (rather than raw network output
    y_probabilities = torch.softmax(y_pred, 1)

    # Print accuracy and log loss
    print("Accuracy:", accuracy_score(y_true, y_pred_classes))
    print("Log loss:", log_loss(y_true, y_probabilities, labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    # Baseline - 1.004
    # evaluate_model(models.BaselineModel())

    # AlexNet Linear - 4.414
    # evaluate_model(
    #     models.AlexNetModel(
    #         state_dict_path="./models/alexnet_2019-10-29_13:35:51.pth", eval_mode=True
    #     )
    # )

    # AlexNet Softmax - 1.338
    # evaluate_model(
    #     models.AlexNetSoftmaxModel(
    #         state_dict_path="./models/alexnet_softmax_2019-10-29_13:51:45.pth",
    #         eval_mode=True,
    #     )
    # )

    # KMeans AlexNet - 25.484
    # _model = models.KMeansModel(
    #     "kmeans_alexnet", model_path="./models/kmeans_alexnet_2019-10-31_13:05:46.pkl"
    # )
    # _feature_extractor = feature_extraction.AlexNet256()
    # evaluate_from_features(_model, _feature_extractor)

    # LDA AlexNet256 - 8.53
    # _model = models.LDAModel(
    #     "lda_alexnet256", model_path="./models/lda_alexnet256_2019-10-31_17:25:25.pkl"
    # )
    # _feature_extractor = features.AlexNet256()
    # evaluate_from_features(_model, _feature_extractor)

    # Basic NN AlexNet256 2019-11-01_17:05:33 - 0.739, 0.672
    # Basic NN AlexNet256 2019-11-01_17:12:59 - 0.764, 0.631
    # Basic NN AlexNet256 2019-11-05_12:45:34 - 0.635, 1.052
    _feature_extractor = features.AlexNet256()
    _model = models.NNModel(
        _feature_extractor.feature_size,
        state_dict_path="./models/basic_nn_2019-11-05_12:45:34.pth",
        eval_mode=True,
    )
    _features_datasets = FeatureDatasets(_feature_extractor)
    evaluate(_model, _features_datasets.get_loader(DatasetType.Test))
