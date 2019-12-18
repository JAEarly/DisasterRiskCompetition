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
from torch.utils.data import DataLoader, ConcatDataset



class PretrainedNNTrainer(ImageTrainer):
    """Pretrained neural network trainer."""

    loss = nn.CrossEntropyLoss

    def __init__(
        self,
        num_epochs=10,
        class_weight_method=ClassWeightMethod.Unweighted,
        root_dir="./data/processed/",
    ):
        super().__init__(root_dir=root_dir)
        self.num_epochs = num_epochs
        self.class_weight_method = class_weight_method

    def train(self, model, train_loader: DataLoader = None, validation_loader: DataLoader = None,  **kwargs) -> (float, float):
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

        # TODO REMOVE - JUST TESTING
        # concat_dataset = ConcatDataset([self.image_datasets.train_dataset, self.image_datasets.validation_dataset, self.image_datasets.test_dataset])
        # train_loader = DataLoader(concat_dataset, batch_size=8, shuffle=True)

        # Setup trial
        trial = Trial(net, optimiser, loss_function, metrics=["loss", "accuracy"]).to(
            device
        )
        trial.with_generators(
            train_loader,
            test_generator=validation_loader,
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
    _network_class = torch_models.resnet152
    _final_layer_alteration = transfers.final_layer_alteration_resnet
    _model = PretrainedNNModel.create_from_transfer(
        _network_class,
        _final_layer_alteration,
        "./models/old_data/grid_search_resnet_custom/all/images_resnet152_2019-12-08_03:49:37.pth",
        3,
        5,
    )
    _trainer = PretrainedNNTrainer(num_epochs=1, root_dir="./data/processed/")
    _trainer.train(_model)
