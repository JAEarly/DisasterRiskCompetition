"""Conv neural network solution."""

import time

import numpy as np
import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial

from models import Model, ClassWeightMethod, ImageTrainer
from utils import class_distribution
import torch.nn.functional as F


class SimpleCNN(torch.nn.Module):

    def __init__(self, num_outputs):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 112 * 112, 64)
        self.fc2 = torch.nn.Linear(64, num_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1, 18 * 112 * 112)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    """Convolutional Neural Network adapted from
    link: @https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    Input images are 224x224 colour rbg (3 channel)
    """
    def __init__(self, num_outputs):  # Is input channel 3 due to RBG?
        super(ConvNet, self).__init__()
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
        self.fc2 = nn.Linear(1000, num_outputs)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class BiggerConvNet(nn.Module):
    """5 convolutional layers and 3 fully connected layers"""
    def __init__(self, num_outputs):  # Is input channel 3 due to RBG?
        super(BiggerConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output will be 16 channels of 112x112 images
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32 Channels of 56X56
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64 channels of 28x28
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128 channels of 14x14
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 256 channels of 7x7
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 265, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_outputs)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class VGGNet(nn.Module):
    """VGG16"""
    def __init__(self, num_outputs):
        super(VGGNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output will be 16 channels of 112x112 images
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32 Channels of 56X56
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64 channels of 28x28
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128 channels of 14x14
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)  # 256 channels of 7x7
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(num_outputs)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)  # check the dimension
        out = self.drop_out(out)  # leave out drop out completely
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class CNNModel(Model):
    def __init__(
        self,
        state_dict_path=None,
        eval_mode=False,
    ):
        super().__init__(str("customcnn").lower())
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Create network and apply final layer alteration to match correct number of classes
        self.net = SimpleCNN(self.num_classes)
        self.net = self.net.to(self.device)
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


class CNNTrainer(ImageTrainer):

    loss = nn.CrossEntropyLoss

    def __init__(
        self, num_epochs=10, class_weight_method=ClassWeightMethod.Unweighted,
    ):
        super().__init__()
        self.num_epochs = num_epochs
        self.class_weight_method = class_weight_method

    def train(self, model: CNNModel, **kwargs) -> (float, float):
        # Get transfer model and put it in training mode
        net = model.net
        net.train()

        # Create optimiser
        optimiser = optim.Adam(net.parameters(), lr=1e-4)

        # Check for cuda
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Training using", device)

        # Setup loss function
        if self.class_weight_method != ClassWeightMethod.Unweighted:
            distribution = class_distribution("data/processed/train")
            if self.class_weight_method == ClassWeightMethod.SumBased:
                inv_distribution = [1 - x / np.sum(distribution) for x in distribution]
                inv_distribution = torch.from_numpy(np.array(inv_distribution)).float()
            elif self.class_weight_method == ClassWeightMethod.MaxBased:
                inv_distribution = [np.max(distribution) / x for x in distribution]
                inv_distribution = torch.from_numpy(np.array(inv_distribution)).float()
            else:
                raise IndexError(
                    "Unknown class weight method " + str(self.class_weight_method)
                )
            loss_function = self.loss(inv_distribution.to(device))
        else:
            loss_function = self.loss()

        # Setup trial
        trial = Trial(net, optimiser, loss_function, metrics=["loss", "accuracy"]).to(
            device
        )
        trial.with_generators(
            self.image_datasets.train_loader,
            test_generator=self.image_datasets.validation_loader,
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
    _model = CNNModel()
    _trainer = CNNTrainer(num_epochs=1)
    _trainer.train(_model)
