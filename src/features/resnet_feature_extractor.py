"""Feature extraction implementation using ResNet."""

import os
import torch
from torch import nn
from torchvision import models

from features import FeatureExtractor, SmoteExtractor, IdentityLayer, DatasetType
from models import cnn_model


class ResNet(FeatureExtractor):
    """ResNet feature extractor."""

    def __init__(self):
        super().__init__("resnet")

    def setup_model(self) -> (nn.Module, int):
        """
        Create a pre-trained ResNet model with the final layer replaced.
        :return: ResNet model.
        """
        resnet = models.resnet152(pretrained=True)  # type: nn.Module
        resnet.fc = IdentityLayer()
        return resnet, 2048


class ResNetSMOTE(SmoteExtractor):
    """ResNet SMOTE feature extractor."""

    def __init__(self):
        super().__init__(ResNet())

    def setup_model(self) -> (nn.Module, int):
        """
        Create a pre-trained ResNet model with the final layer replaced.
        :return: ResNet model.
        """
        resnet = models.resnet152(pretrained=True)  # type: nn.Module
        resnet.fc = IdentityLayer()
        return resnet, 2048


class ResNetCustom(FeatureExtractor):
    """AlexNet feature extractor using a custom trained model."""

    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        print("Creating ResNet from", model_path)
        super().__init__("resnet_custom")

    def setup_model(self) -> (nn.Module, int):
        resnet = models.resnet152()
        resnet = cnn_model.final_layer_alteration_resnet(resnet, 5)
        resnet.load_state_dict(torch.load(self.model_path))
        resnet.fc = IdentityLayer()
        resnet.eval()
        return resnet, 2048


if __name__ == "__main__":
    print("Creating ResNet extractor")
    feature_extractor = ResNet()
    print("Extracting features")
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
    feature_extractor.extract(DatasetType.Competition)

    print("Creating ResNet SMOTE extractor")
    feature_extractor = ResNetSMOTE()
    print("Extracting features")
    feature_extractor.extract(DatasetType.Train)

    print("Creating ResNet custom extractor")
    feature_extractor = ResNetCustom("./models/grid_search_resnet_cnn/best.pth")
    print("Extracting features")
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
    feature_extractor.extract(DatasetType.Competition)
