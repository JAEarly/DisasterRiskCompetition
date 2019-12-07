import time

import numpy as np
import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial
from torchvision import models as torch_models

from models import transfers, PretrainedNNModel
from training import ClassWeightMethod, ImageTrainer
from utils import class_distribution


class PretrainedNNTrainer(ImageTrainer):
    """Pretrained neural network trainer."""

    loss = nn.CrossEntropyLoss

    def __init__(
        self, num_epochs=10, class_weight_method=ClassWeightMethod.Unweighted, train_dir="./data/processed/train"
    ):
        super().__init__(train_dir=train_dir)
        self.num_epochs = num_epochs
        self.class_weight_method = class_weight_method

    def train(self, model, **kwargs) -> (float, float):
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
    _network_class = torch_models.vgg19_bn
    _model = PretrainedNNModel(_network_class, transfers.final_layer_alteration_alexnet)
    _trainer = PretrainedNNTrainer(num_epochs=1)
    _trainer.train(_model)
