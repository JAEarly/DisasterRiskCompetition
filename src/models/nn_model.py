"""Neural network solution."""

import torch
import torch.nn.functional as F
from torch import nn

from models import Model


class LinearNN(nn.Module):
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

    def __init__(self, input_size, output_size, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class BiggerConvNet(nn.Module):
    """Convolutional Neural Network adapted from
    link: @https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    Input images are 224x224 colour rbg (3 channel)
    """
    def __init__(self):  # Is input channel 3 due to RBG?
        super(BiggerConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output will be 32 channels of 112x112 images
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 64 Channels of 56X56
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128 channels of 28x28
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(28 * 28 * 128, 1000)
        self.fc2 = nn.Linear(1000, 5)


class AlexNetClassifierNN(nn.Module):
    """AlexNet classifier layer NN implementation."""

    def __init__(self, input_size, output_size, dropout=0):
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
        super().__init__(str(net_class.__name__).lower(), True)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Create network
        self.net = net_class(input_size, self.num_classes, dropout=dropout).to(
            self.device
        )
        # Load network state if provided
        if state_dict_path is not None:
            self.load(state_dict_path)
        if eval_mode:
            self.net.eval()

    def predict(self, feature_tensor):
        return self.net(feature_tensor.to(self.device))

    def predict_batch(self, feature_batch):
        return self.net(feature_batch.to(self.device))

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.net.state_dict(), path)
