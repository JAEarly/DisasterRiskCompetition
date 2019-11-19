"""Conv neural network solution."""

import time

import numpy as np
import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial
from torchvision import models as torch_models

from models import Model, ClassWeightMethod, ImageTrainer
from utils import class_distribution


def final_layer_alteration_alexnet(net, num_classes):
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


class PretrainedNNModel(Model):
    """Base model that uses a pretrained cnn."""

    def __init__(
        self,
        pretrained_net_class,
        final_layer_alteration,
        state_dict_path=None,
        eval_mode=False,
    ):
        super().__init__(str(pretrained_net_class.__name__).lower())
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Create network and apply final layer alteration to match correct number of classes
        self.net = pretrained_net_class(pretrained=True)
        self.net = final_layer_alteration(self.net, self.num_classes)
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


class PretrainedNNTrainer(ImageTrainer):
    """Pretrained neural network trainer."""

    loss = nn.CrossEntropyLoss

    def __init__(
        self, num_epochs=10, class_weight_method=ClassWeightMethod.Unweighted,
    ):
        super().__init__()
        self.num_epochs = num_epochs
        self.class_weight_method = class_weight_method

    def train(self, model: PretrainedNNModel, **kwargs) -> (float, float):
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
    _network_class = torch_models.alexnet
    _model = PretrainedNNModel(_network_class, final_layer_alteration_alexnet)
    _trainer = PretrainedNNTrainer(num_epochs=1)
    _trainer.train(_model)
