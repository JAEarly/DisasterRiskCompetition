import os
import time

import numpy as np
import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial

from features import FeatureExtractor, AlexNet256
from models import FeatureTrainer, NNModel
from utils import create_timestamp_str, class_distribution


class NNTrainer(FeatureTrainer):

    num_epochs = 10
    loss = nn.CrossEntropyLoss

    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__(feature_extractor)

    def train(self, model: NNModel, class_weights=None):
        # Get transfer model and put it in training mode
        net = model.net
        net.train()

        # Create optimiser for active parameters only
        optimiser = optim.Adam(net.parameters(), lr=1e-4)

        # Setup loss function
        if class_weights is not None:
            loss_function = self.loss(weight=class_weights)
        else:
            loss_function = self.loss()

        # Setup trial
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        trial = Trial(net, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
        trial.with_generators(self.feature_dataset.train_loader, test_generator=self.feature_dataset.test_loader)

        # Actually run the training
        trial.run(epochs=self.num_epochs)

        # Evaluate and show results
        time.sleep(1)  # Ensure training has finished
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)
        print(results)

        # Save model weights
        save_path = os.path.join(self.save_dir, model.name + "_" + create_timestamp_str() + ".pth")
        model.save(save_path)


if __name__ == "__main__":
    trainer = NNTrainer(AlexNet256())
    model = NNModel(trainer.feature_dataset.feature_size)
    class_distribution = class_distribution("data/processed/train")
    _class_weights = [1 - x/sum(class_distribution) for x in class_distribution]
    _class_weights = torch.from_numpy(np.array(_class_weights)).float()
    trainer.train(model, class_weights=_class_weights)
