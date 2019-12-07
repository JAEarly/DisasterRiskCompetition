"""Feature extraction implementation using ResNet."""

import os
import torch
from torch import nn
from torchvision import models

import models.transfers as transfers
from features import FeatureExtractor, SmoteExtractor, IdentityLayer, DatasetType
from features.smote_extractor import smote_type_to_name, SmoteType

DEFAULT_CUSTOM_PATH = "./models/verified/grid_search_resnet_custom/best.pth"


def setup_resnet():
    resnet = models.resnet152(pretrained=True)  # type: nn.Module
    resnet.fc = IdentityLayer()
    resnet.eval()
    return resnet


def setup_resnet_custom(model_path):
    # Create ResNet model from custom pretrained state.
    #  Must initially alter the final layer to match architectures.
    resnet = models.resnet152()
    resnet = transfers.final_layer_alteration_resnet(resnet, 5)
    resnet.load_state_dict(torch.load(model_path))
    # Now replace final layer with an identity layer
    resnet.fc = IdentityLayer()
    resnet.eval()
    return resnet


class ResNet(FeatureExtractor):
    """ResNet feature extractor."""

    def __init__(self, save_dir="./models/features/", train_dir="./data/processed/train"):
        super().__init__("resnet", save_dir=save_dir, train_dir=train_dir)

    def setup_model(self) -> (nn.Module, int):
        """
        Create a pre-trained ResNet model with the final layer replaced.
        :return: ResNet model.
        """
        resnet = setup_resnet()
        return resnet, 2048


class ResNetSMOTE(SmoteExtractor):
    """ResNet SMOTE feature extractor."""

    def __init__(self, save_dir="./models/features/", train_dir="./data/processed/train"):
        super().__init__(ResNet(), save_dir=save_dir, train_dir=train_dir)

    def setup_model(self) -> (nn.Module, int):
        """
        Create a pre-trained ResNet model with the final layer replaced.
        :return: ResNet model.
        """
        resnet = setup_resnet()
        return resnet, 2048


class ResNetCustom(FeatureExtractor):
    """ResNet feature extractor using a custom trained model."""

    def __init__(self, model_path=DEFAULT_CUSTOM_PATH, save_dir="./models/features/", train_dir="./data/processed/train"):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        super().__init__("resnet_custom", save_dir=save_dir, train_dir=train_dir)

    def setup_model(self) -> (nn.Module, int):
        resnet = setup_resnet_custom(self.model_path)
        return resnet, 2048


class ResNetCustomSMOTE(SmoteExtractor):
    """ResNet feature extractor using a custom trained model with SMOTE."""

    def __init__(self, model_path=DEFAULT_CUSTOM_PATH, smote_type: SmoteType = SmoteType.Normal, save_dir="./models/features/", train_dir="./data/processed/train"):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        super().__init__(ResNetCustom(model_path), smote_type=smote_type, save_dir=save_dir, train_dir=train_dir)

    def setup_model(self) -> (nn.Module, int):
        resnet = setup_resnet_custom(self.model_path)
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
    #
    print("Creating ResNet custom extractor")
    feature_extractor = ResNetCustom()
    print("Extracting features")
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
    feature_extractor.extract(DatasetType.Competition)

    for _smote_type in SmoteType:
        print("Creating ResNet custom", smote_type_to_name(_smote_type), "extractor")
        feature_extractor = ResNetCustomSMOTE(smote_type=_smote_type)
        print("Extracting features")
        feature_extractor.extract(DatasetType.Train)
