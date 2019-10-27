import torch
from torch import nn
from torchvision import models

from .transfer_model import TransferModel


class AlexNetModel(TransferModel):

    def __init__(self, state_dict_path=None, eval_mode=False):
        super().__init__("alexnet", state_dict_path=state_dict_path, eval_mode=eval_mode)

    def setup_transfer_model(self, state_dict_path, eval_mode):
        # Create AlexNet model with linear output layer
        linear_alexnet_model = models.alexnet(pretrained=True)
        linear_alexnet_model.classifier = nn.Linear(9216, self.num_classes)

        # Keep all layers fixed except final layer
        for param in linear_alexnet_model.parameters():
            param.requires_grad = False
        linear_alexnet_model.classifier.weight.requires_grad = True
        linear_alexnet_model.classifier.bias.requires_grad = True

        # Load existing weights if given
        if state_dict_path is not None:
            linear_alexnet_model.load_state_dict(torch.load(state_dict_path))

        # Put into evaluation mode if required
        if eval_mode:
            linear_alexnet_model.eval()

        return linear_alexnet_model
