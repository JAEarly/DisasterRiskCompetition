import os

import models
from models import Trainer, KMeansModel, FeatureExtractor, AlexNet256
from utils import create_timestamp_str


class KMeansTrainer(Trainer):

    def __init__(self, model: KMeansModel, feature_extractor: FeatureExtractor):
        super().__init__(model)
        self.feature_extractor = feature_extractor

    def train(self, class_weights=None):
        print("Loading features")
        training_features = self.feature_extractor.extract_features(self.train_loader, "train")
        print("Fitting model")
        self.model.fit(training_features)
        print("Saving model")
        save_path = os.path.join(self.save_dir, self.model.name + "_" + create_timestamp_str() + ".pkl")
        self.model.save(save_path)


if __name__ == "__main__":
    print("Creating KMeans model")
    kmeans_model = models.KMeansModel("kmeans_alexnet")
    print("Creating AlexNet feature extractor")
    trainer = KMeansTrainer(kmeans_model, AlexNet256())
    print("Training...")
    trainer.train()
