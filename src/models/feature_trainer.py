from abc import ABC

from features import FeatureDatasets, FeatureExtractor
from models import Trainer


class FeatureTrainer(Trainer, ABC):

    # TODO feature model
    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__()
        self.feature_dataset = FeatureDatasets(feature_extractor)
