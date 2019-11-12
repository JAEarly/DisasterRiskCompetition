"""Feature extraction implementation using ResNet."""

from torch import nn
from torchvision import models
from torchvision import transforms

from features import FeatureExtractor, IdentityLayer, DatasetType


class ResNet18t256(FeatureExtractor):
    """ResNet feature extractor using a 256 image transform."""

    def __init__(self):
        super().__init__("resnet18t256")

    def setup_model(self) -> (nn.Module, int):
        """
        Create a pre-trained ResNet model with the final layer replaced.
        :return: ResNet model.
        """
        resnet = models.resnet18(pretrained=True)  # type: nn.Module
        resnet.fc = IdentityLayer()
        return resnet, 512

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
    print("Creating ResNet18t256 extractor")
    feature_extractor = ResNet18t256()
    print("Extracting features")
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
    feature_extractor.extract(DatasetType.Competition)
