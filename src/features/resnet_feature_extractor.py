"""Feature extraction implementation using ResNet."""

from torch import nn
from torchvision import models

from features import FeatureExtractor, IdentityLayer, DatasetType, SmoteExtractor


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
