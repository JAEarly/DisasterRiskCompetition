import os

import models
from models import Trainer, KMeansModel
from features import FeatureExtractor, AlexNet256, DatasetType
from utils import create_timestamp_str


class LDATrainer(Trainer):

    def __init__(self, model: models.LDAModel, feature_extractor: FeatureExtractor):
        super().__init__(model)
        self.feature_extractor = feature_extractor

    def train(self, class_weights=None):
        print("Loading features")
        features, labels = self.feature_extractor.extract(DatasetType.Train)
        print("Fitting model")
        self.model.fit(features, labels)
        print("Saving model")
        save_path = os.path.join(self.save_dir, self.model.name + "_" + create_timestamp_str() + ".pkl")
        self.model.save(save_path)


if __name__ == "__main__":
    print("Creating LDA model")
    lda_model = models.LDAModel("lda_alexnet256")
    print("Creating AlexNet feature extractor")
    trainer = LDATrainer(lda_model, AlexNet256())
    print("Training...")
    trainer.train()
