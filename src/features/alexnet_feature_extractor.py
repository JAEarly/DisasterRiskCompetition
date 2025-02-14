"""Feature extraction implementation using AlexNet."""

import os
import torch
from torch import nn
from torchvision import models

import models.transfers as transfers
from features import FeatureExtractor, SmoteExtractor, IdentityLayer, DatasetType

DEFAULT_CUSTOM_PATH = "./models/grid_search_alexnet_custom/best.pth"


def setup_alexnet():
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier = IdentityLayer()
    alexnet.eval()
    return alexnet


def setup_alexnet_custom(model_path: str):
    # Create AlexNet model from custom pretrained state.
    #  Must initially alter the final layer to match architectures.
    alexnet = models.alexnet()
    alexnet = transfers.final_layer_alteration_alexnet(alexnet, 5)
    alexnet.load_state_dict(torch.load(model_path))

    # Now replace final layer with an identity layer
    alexnet.classifier = IdentityLayer()
    alexnet.eval()
    return alexnet


class AlexNet(FeatureExtractor):
    """AlexNet feature extractor using a 256 image transform."""

    def __init__(self):
        super().__init__("alexnet")

    def setup_model(self) -> (nn.Module, int):
        """
        Create a pre-trained AlexNet model with the final layer replace.
        :return: AlexNet model.
        """
        alexnet = setup_alexnet()
        return alexnet, 9216


class AlexNetSMOTE(SmoteExtractor):
    """AlexNet SMOTE feature extractor."""

    def __init__(self):
        super().__init__(AlexNet())

    def setup_model(self) -> (nn.Module, int):
        """
        Create a pre-trained AlexNet model with the final layer replace.
        :return: AlexNet model.
        """
        alexnet = setup_alexnet()
        return alexnet, 9216


class AlexNetCustom(FeatureExtractor):
    """AlexNet feature extractor using a custom trained model."""

    def __init__(self, model_path=DEFAULT_CUSTOM_PATH):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        super().__init__("alexnet_custom")

    def setup_model(self) -> (nn.Module, int):
        alexnet = setup_alexnet_custom(self.model_path)
        return alexnet, 9216


class AlexNetCustomSMOTE(SmoteExtractor):
    """AlexNet feature extractor using a custom trained model with SMOTE."""

    def __init__(self, model_path=DEFAULT_CUSTOM_PATH):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        super().__init__(AlexNetCustom(model_path))

    def setup_model(self) -> (nn.Module, int):
        alexnet = setup_alexnet_custom(self.model_path)
        return alexnet, 9216


if __name__ == "__main__":
    print("Creating AlexNet extractor")
    feature_extractor = AlexNet()
    print("Extracting")
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
    feature_extractor.extract(DatasetType.Competition)

    print("Creating AlexNet SMOTE extractor")
    feature_extractor = AlexNetSMOTE()
    print("Extracting")
    feature_extractor.extract(DatasetType.Train)

    print("Creating custom AlexNet extractor")
    feature_extractor = AlexNetCustom()
    print("Extracting")
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
    feature_extractor.extract(DatasetType.Competition)

    print("Creating AlexNet Custom SMOTE extractor")
    feature_extractor = AlexNetCustomSMOTE()
    print("Extracting")
    feature_extractor.extract(DatasetType.Train)
