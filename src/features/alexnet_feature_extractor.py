from features import FeatureExtractor, IdentityLayer, DatasetType
from torchvision import models
from torchvision import transforms


class AlexNet256(FeatureExtractor):

    def __init__(self):
        super().__init__("alexnet")

    def setup_model(self):
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier = IdentityLayer()
        return alexnet

    def get_transform(self):
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()])
        return transform


if __name__ == "__main__":
    print("Creating AlexNet256 extractor")
    feature_extractor = AlexNet256()
    feature_extractor.extract(DatasetType.Train)
    feature_extractor.extract(DatasetType.Validation)
    feature_extractor.extract(DatasetType.Test)
