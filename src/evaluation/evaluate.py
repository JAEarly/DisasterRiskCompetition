from sklearn.metrics import log_loss, accuracy_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import models
import features
from features import DatasetType
import torch

TEST_DATA_DIR = "./data/processed/test"
BATCH_SIZE = 8


def evaluate_model(model: models.Model):
    dataset = ImageFolder(TEST_DATA_DIR, transform=model.get_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true = []
    y_pred = []

    for batch, labels in tqdm(dataloader):
        y_true.extend(labels)
        y_pred.extend(model.predict_batch(batch))

    print("Log loss:", log_loss(y_true, y_pred, labels=[0, 1, 2, 3, 4]))


def evaluate_from_features(model: models.Model, feature_extractor: features.feature_extractor):
    test_features, labels = feature_extractor.extract(DatasetType.Test)

    y_true = []
    y_pred = []

    for feature, label in tqdm(zip(test_features, labels), total=len(labels)):
        y_pred.append(model.predict(feature.unsqueeze(0))[0])
        y_true.append(label)

    y_pred = torch.stack(y_pred).cpu().detach()
    _, y_pred_classes = y_pred.max(1)

    print("Accuracy:", accuracy_score(y_true, y_pred_classes))
    print("Log loss:", log_loss(y_true, y_pred, labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    # Baseline - 1.004
    # evaluate_model(models.BaselineModel())

    # AlexNet Linear - 4.414
    # evaluate_model(models.AlexNetModel(state_dict_path="./models/alexnet_2019-10-29_13:35:51.pth", eval_mode=True))

    # AlexNet Softmax - 1.338
    # evaluate_model(models.AlexNetSoftmaxModel(state_dict_path="./models/alexnet_softmax_2019-10-29_13:51:45.pth", eval_mode=True))

    # KMeans AlexNet - 25.484
    # _model = models.KMeansModel("kmeans_alexnet", model_path="./models/kmeans_alexnet_2019-10-31_13:05:46.pkl")
    # _feature_extractor = feature_extraction.AlexNet256()
    # evaluate_from_features(_model, _feature_extractor)

    # LDA AlexNet256 - 8.53
    # _model = models.LDAModel("lda_alexnet256", model_path="./models/lda_alexnet256_2019-10-31_17:25:25.pkl")
    # _feature_extractor = features.AlexNet256()
    # evaluate_from_features(_model, _feature_extractor)

    # Basic NN AlexNet256 2019-11-01_12:54:08 (1 epoch)   - 0.742, 1.585
    # Basic NN AlexNet256 2019-11-01_13:04:56 (5 epochs)  - 0.766, 1.794
    # Basic NN AlexNet256 2019-11-01_13:15:26 (15 epochs) - 0.776, 2.882
    _model = models.NNModel(9216, state_dict_path="./models/basic_nn_2019-11-01_13:15:26.pth", eval_mode=True)
    _feature_extractor = features.AlexNet256()
    evaluate_from_features(_model, _feature_extractor)
