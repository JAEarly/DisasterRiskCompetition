import time

import torch
import torchbearer
from torch import optim
from torch.utils.data import DataLoader
from torchbearer import Trial

import features
import models
from features import BalanceMethod, FeatureExtractor
from models import NNModel
from training import FeatureTrainer, ClassWeightMethod, SmoothedCrossEntropyLoss


class NNTrainer(FeatureTrainer):
    """Neural network trainer."""

    loss = SmoothedCrossEntropyLoss

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        balance_method=BalanceMethod.NoSample,
        num_epochs=10,
        class_weight_method=ClassWeightMethod.Unweighted,
        label_smoothing=0,
    ):
        super().__init__(feature_extractor, balance_method=balance_method)
        self.num_epochs = num_epochs
        self.class_weight_method = class_weight_method
        self.label_smoothing = label_smoothing

    def train(
        self, model, train_loader: DataLoader = None, validation_loader: DataLoader = None, **kwargs
    ) -> (float, float):
        if train_loader is None:
            train_loader = self.feature_dataset.train_loader
        if validation_loader is None:
            validation_loader = self.feature_dataset.validation_loader

        # Get transfer model and put it in training mode
        net = model.net
        net.train()

        # Create optimiser
        optimiser = optim.Adam(net.parameters(), lr=1e-4)

        # Check for cuda
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Training using", device)

        # Setup loss function
        loss_function = SmoothedCrossEntropyLoss(model.num_classes, smoothing=self.label_smoothing)

        # Setup trial
        trial = Trial(net, optimiser, loss_function, metrics=["loss", "accuracy"]).to(
            device
        )
        trial.with_generators(
            train_loader, test_generator=validation_loader,
        )

        # Actually run the training
        trial.run(epochs=self.num_epochs)

        # Evaluate and show results
        time.sleep(0.1)  # Ensure training has finished
        net.eval()
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)

        acc = float(results["test_acc"])
        loss = float(results["test_loss"])
        return acc, loss


if __name__ == "__main__":
    _network_class = models.BiggerNN
    _feature_extractor = features.AlexNet()
    _trainer = NNTrainer(_feature_extractor, balance_method=BalanceMethod.OverSample)
    _model = NNModel(_network_class, _feature_extractor.feature_size)
    _trainer.train(_model)
