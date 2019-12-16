import time

import torch
import torchbearer
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from torchbearer import Trial

import features
import models
from features import BalanceMethod, FeatureExtractor
from models import NNModel
from training import Trainer, FeatureTrainer, ClassWeightMethod, SmoothedPseudoCrossEntropyLoss
from torch.nn.modules import CrossEntropyLoss


class NNTrainer(FeatureTrainer):
    """Neural network trainer."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        balance_method=BalanceMethod.NoSample,
        num_epochs=10,
        class_weight_method=ClassWeightMethod.Unweighted,
        label_smoothing=0,
        alpha=0,
    ):
        super().__init__(feature_extractor, balance_method=balance_method)
        self.num_epochs = num_epochs
        self.class_weight_method = class_weight_method
        self.label_smoothing = label_smoothing
        self.alpha = alpha

    def train(
        self, model, train_loader: DataLoader = None, validation_loader: DataLoader = None, **kwargs
    ) -> (float, float):
        if train_loader is None:
            train_loader = self.feature_dataset.train_loader
        if validation_loader is None:
            validation_loader = self.feature_dataset.validation_loader

        # TODO REMOVE - JUST TESTING
        # concat_dataset = ConcatDataset([self.feature_dataset.train_dataset, self.feature_dataset.validation_dataset, self.feature_dataset.test_dataset])
        # train_loader = DataLoader(concat_dataset, batch_size=8, shuffle=True)

        # Get transfer model and put it in training mode
        net = model.net
        net.train()

        # Create optimiser
        optimiser = optim.Adam(net.parameters(), lr=1e-4)

        # Check for cuda
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Training using", device)

        # Setup loss function
        loss_function = SmoothedPseudoCrossEntropyLoss(self.feature_dataset.pseudo_loader, model, smoothing=self.label_smoothing, alpha=self.alpha)

        # Setup trial
        trial = Trial(net, optimiser, loss_function, metrics=["loss", "accuracy"]).to(
            device
        )
        trial.with_generators(
            train_loader
        )

        # Actually run the training
        trial.run(epochs=self.num_epochs)

        # Evaluate and show results
        time.sleep(0.1)  # Ensure training has finished
        net.eval()
        acc, loss = Trainer.evaluate(model, validation_loader)
        return acc, loss


if __name__ == "__main__":
    _network_class = models.BiggerNN
    _feature_extractor = features.AlexNet()
    _trainer = NNTrainer(_feature_extractor, balance_method=BalanceMethod.OverSample)
    _model = NNModel(_network_class, _feature_extractor.feature_size)
    _trainer.train(_model)
