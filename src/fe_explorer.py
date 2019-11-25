from torchvision import models as tv_models

import models
import models.transfers as transfers
from features import DatasetType, FeatureDatasets, ImageDatasets
from features import ResNetCustom
from models import FeatureTrainer
from models import NNModel
from models.cnn_model import PretrainedNNTrainer

feature_extractor = ResNetCustom("./models/grid_search_resnet_cnn/best.pth")

print("Loading CNN Model")
cnn_model = models.PretrainedNNModel(
    tv_models.resnet152,
    transfers.final_layer_alteration_resnet,
    state_dict_path="./models/grid_search_resnet_cnn/best.pth",
    eval_mode=True,
)

print("Loading Linear Model")
lnn_model = NNModel(models.LinearNN, feature_extractor.feature_size, eval_mode=True)

print("Setting fc weights")
lnn_model.net.fc1.weight = cnn_model.net.fc.weight
lnn_model.net.fc1.bias = cnn_model.net.fc.bias

print("Evaluating CNN:")
image_datasets = ImageDatasets()
PretrainedNNTrainer.evaluate(
    cnn_model,
    image_datasets.get_loader(DatasetType.Train),
    apply_softmax=True,
    verbose=True,
)

print("Evaluating LNN:")
features_datasets = FeatureDatasets(feature_extractor)
FeatureTrainer.evaluate(
    lnn_model,
    features_datasets.get_loader(DatasetType.Train),
    apply_softmax=True,
    verbose=True,
)
