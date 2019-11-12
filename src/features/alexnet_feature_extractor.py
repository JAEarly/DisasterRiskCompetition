"""Feature extraction implementation using AlexNet."""

from torch import nn
from torchvision import models
from torchvision import transforms

from features import FeatureExtractor, IdentityLayer, DatasetType


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


if __name__ == "__main__":
    print("Creating AlexNet extractor")
    feature_extractor = AlexNet()
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
    feature_extractor.extract(DatasetType.Competition)
