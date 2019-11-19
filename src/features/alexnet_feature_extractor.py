"""Feature extraction implementation using AlexNet."""

from torch import nn
from torchvision import models

from features import FeatureExtractor, SmoteExtractor, IdentityLayer, DatasetType


class AlexNet(FeatureExtractor):
    """AlexNet feature extractor using a 256 image transform."""

    def __init__(self):
        super().__init__("alexnet")

    def setup_model(self) -> (nn.Module, int):
        """
        Create a pre-trained AlexNet model with the final layer replace.
        :return: AlexNet model.
        """
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier = IdentityLayer()
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
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier = IdentityLayer()
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
