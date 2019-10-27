import os
import time

import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial

from models import TransferModel, AlexNetModel, Trainer


class TransferTrainer(Trainer):

    batch_size = 8
    num_epochs = 1
    loss_function = nn.CrossEntropyLoss()

    def __init__(self, model: TransferModel):
        super().__init__(model)

    def train(self):
        # Get transfer model and put it in training mode
        transfer_model = self.model.transfer_model
        transfer_model.train()

        # Create optimiser for active parameters only
        optimiser = optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=1e-4)

        # Setup trial
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        trial = Trial(transfer_model, optimiser, self.loss_function, metrics=['loss', 'accuracy']).to(device)
        trial.with_generators(self.train_loader, test_generator=self.test_loader)

        # Actually run the training
        trial.run(epochs=self.num_epochs)

        # Evaluate and show results
        time.sleep(1)  # Ensure training has finished
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)
        print(results)

        # Save model weights
        # TODO timestamp and find appropriate file extension
        save_path = os.path.join(self.save_dir, self.model.name + ".model")
        self.model.save(save_path)


if __name__ == "__main__":
    alexnet_model = AlexNetModel()
    trainer = TransferTrainer(alexnet_model)
    trainer.train()
