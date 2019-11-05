"""Feature extraction implementation using AlexNet."""

from torch import nn
from torchvision import models
from torchvision import transforms

from features import FeatureExtractor, IdentityLayer, DatasetType


class AlexNet256(FeatureExtractor):
    """AlexNet feature extractor using a 256 image transform."""

    def __init__(self):
        super().__init__("alexnet")

    def setup_model(self) -> nn.Module:
        """
        Create a pre-trained AlexNet model with the final layer replace.
        :return: AlexNet model.
        """
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier = IdentityLayer()
        return alexnet

    def get_transform(self) -> transforms.Compose:
        """
        Create a transform that reduces to images to 256 x 256.
        :return: Composed transform.
        """
        transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )
        return transform


if __name__ == "__main__":
    print("Creating AlexNet256 extractor")
    feature_extractor = AlexNet256()
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
    feature_extractor.extract(DatasetType.Competition)
