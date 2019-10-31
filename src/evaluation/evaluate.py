from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import models
import models.feature_extraction as feature_extraction

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
    features, labels = feature_extractor.extract(feature_extractor.test_loader, "test")

    y_true = []
    y_pred = []

    for feature, label in zip(features, labels):
        y_pred.append(model.predict(feature.unsqueeze(0)))
        y_true.append(label)

    print("Log loss:", log_loss(y_true, y_pred, labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    # Baseline - 1.004
    # evaluate_model(models.BaselineModel())

    # AlexNet Linear - 4.414
    # evaluate_model(models.AlexNetModel(state_dict_path="./models/alexnet_2019-10-29_13:35:51.pth", eval_mode=True))

    # AlexNet Softmax - 1.338
    #evaluate_model(models.AlexNetSoftmaxModel(state_dict_path="./models/alexnet_softmax_2019-10-29_13:51:45.pth", eval_mode=True))

    # KMeans AlexNet - 25.484
    _model = models.KMeansModel("kmeans_alexnet", model_path="./models/kmeans_alexnet_2019-10-31_13:05:46.pkl")
    _feature_extractor = feature_extraction.AlexNet256()
    evaluate_from_features(_model, _feature_extractor)
