"""Neural network solution."""

import time

import numpy as np
import torch
import torch.nn.functional as F
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial

import features
from features import BalanceMethod, FeatureExtractor
from models import FeatureTrainer, Model, ClassWeightMethod
from utils import class_distribution


class LinearNN(nn.Module):
    """Linear NN implementation."""

    def __init__(self, input_size, output_size, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x


class LinearNNWithDropout(nn.Module):
    """Linear NN implementation."""

    def __init__(self, input_size, output_size, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        return x


class BiggerNN(nn.Module):
    """Bigger NN implementation."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class AlexNetClassifierNN(nn.Module):
    """AlexNet classifier layer NN implementation."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_size),
        )

    def forward(self, x):
        return self.classifier(x)


class NNModel(Model):
    """Base model that uses a neural network."""

    def __init__(
        self,
        net_class,
        input_size: int,
        state_dict_path=None,
        eval_mode=False,
        dropout=0,
    ):
        super().__init__(str(net_class.__name__).lower())
        # Create network
        self.net = net_class(input_size, self.num_classes, dropout=dropout)
        # Load network state if provided
        if state_dict_path is not None:
            self.load(state_dict_path)
        if eval_mode:
            self.net.eval()

    def predict(self, feature_tensor):
        return self.net(feature_tensor)

    def predict_batch(self, feature_batch):
        return self.net(feature_batch)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.net.state_dict(), path)


class NNTrainer(FeatureTrainer):
    """Neural network trainer."""

    loss = nn.CrossEntropyLoss

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        balance_method=BalanceMethod.NoSample,
        num_epochs=10,
        class_weight_method=ClassWeightMethod.Unweighted,
    ):
        super().__init__(feature_extractor, balance_method)
        self.num_epochs = num_epochs
        self.class_weight_method = class_weight_method

    def train(self, model: NNModel) -> (float, float):
        # Get transfer model and put it in training mode
        net = model.net
        net.train()

        # Create optimiser
        optimiser = optim.Adam(net.parameters(), lr=1e-4)

        # Setup loss function
        if self.class_weight_method != ClassWeightMethod.Unweighted:
            distribution = class_distribution("data/processed/train")
            if self.class_weight_method == ClassWeightMethod.SumBased:
                inv_distribution = [1 - x / np.sum(distribution) for x in distribution]
                inv_distribution = torch.from_numpy(np.array(inv_distribution)).float()
            elif self.class_weight_method == ClassWeightMethod.MaxBased:
                inv_distribution = [np.max(distribution) / x for x in distribution]
                inv_distribution = torch.from_numpy(np.array(inv_distribution)).float()
            loss_function = self.loss(inv_distribution)
        else:
            loss_function = self.loss()

        # Setup trial
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        trial = Trial(net, optimiser, loss_function, metrics=["loss", "accuracy"]).to(
            device
        )
        trial.with_generators(
            self.feature_dataset.train_loader,
            test_generator=self.feature_dataset.test_loader,
        )

        # Actually run the training
        trial.run(epochs=self.num_epochs)

        # Evaluate and show results
        time.sleep(0.1)  # Ensure training has finished
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)
        acc = float(results["test_acc"])
        loss = float(results["test_loss"])

        return acc, loss


if __name__ == "__main__":
    _network_class = BiggerNN
    _feature_extractor = features.AlexNet()
    _trainer = NNTrainer(_feature_extractor, balance_method=BalanceMethod.OverSample)
    _model = NNModel(_network_class, _feature_extractor.feature_size)
    _trainer.train(_model)
