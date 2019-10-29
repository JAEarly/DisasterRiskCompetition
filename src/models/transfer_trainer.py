import os
import time

import numpy as np
import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial

import models
from models import TransferModel, Trainer
from utils import create_timestamp_str, class_distribution


class TransferTrainer(Trainer):

    batch_size = 8
    num_epochs = 1
    loss = nn.CrossEntropyLoss

    def __init__(self, model: TransferModel):
        super().__init__(model)

    def train(self, class_weights=None):
        # Get transfer model and put it in training mode
        transfer_model = self.model.transfer_model
        transfer_model.train()

        # Create optimiser for active parameters only
        optimiser = optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=1e-4)

        # Setup loss function
        if class_weights is not None:
            loss_function = self.loss(weight=class_weights)
        else:
            loss_function = self.loss()

        # Setup trial
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        trial = Trial(transfer_model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
        trial.with_generators(self.train_loader, test_generator=self.test_loader)

        # Actually run the training
        trial.run(epochs=self.num_epochs)

        # Evaluate and show results
        time.sleep(1)  # Ensure training has finished
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)
        print(results)

        # Save model weights
        save_path = os.path.join(self.save_dir, self.model.name + "_" + create_timestamp_str() + ".pth")
        self.model.save(save_path)


if __name__ == "__main__":
    alexnet_model = models.AlexNetSoftmaxModel()
    trainer = TransferTrainer(alexnet_model)
    class_distribution = class_distribution("data/processed/train")
    _class_weights = [1 - x/sum(class_distribution) for x in class_distribution]
    _class_weights = torch.from_numpy(np.array(_class_weights)).float()
    trainer.train(class_weights=_class_weights)
