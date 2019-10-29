import torch
from torch import nn
from torchvision import models

from .transfer_model import TransferModel


class AlexNetSoftmaxModel(TransferModel):

    def __init__(self, state_dict_path=None, eval_mode=False):
        super().__init__("alexnet_softmax", state_dict_path=state_dict_path, eval_mode=eval_mode)

    def setup_transfer_model(self, state_dict_path, eval_mode):
        # Create AlexNet model with softmax output layer
        softmax_alexnet_model = models.alexnet(pretrained=True)
        softmax_alexnet_model.classifier = nn.Sequential(nn.Linear(9216, self.num_classes), nn.Softmax(1))

        # Keep all layers fixed except final layer
        for param in softmax_alexnet_model.parameters():
            param.requires_grad = False
        softmax_alexnet_model.classifier[0].weight.requires_grad = True
        softmax_alexnet_model.classifier[0].bias.requires_grad = True

        # Load existing weights if given
        if state_dict_path is not None:
            softmax_alexnet_model.load_state_dict(torch.load(state_dict_path))

        # Put into evaluation mode if required
        if eval_mode:
            softmax_alexnet_model.eval()

        return softmax_alexnet_model
