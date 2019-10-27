import torch
from torch import nn
from torchvision import models

from .model import Model


class AlexNetModel(Model):

    num_classes = 5

    def __init__(self, state_dict_path=None, eval_mode=False):
        super().__init__("alexnet")

        # Create AlexNet model setup for this task
        self.cnn_model = models.alexnet(pretrained=True)
        self.cnn_model.classifier = nn.Linear(9216, self.num_classes)

        # Fix all layers except final layer
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        self.cnn_model.classifier.weight.requires_grad = True
        self.cnn_model.classifier.bias.requires_grad = True

        # Load existing weights if given
        if state_dict_path is not None:
            self.cnn_model.load_state_dict(torch.load(state_dict_path))

        # Put into evaluation mode if request
        if eval_mode:
            self.cnn_model.eval()

    def predict(self, image_tensor):
        return self.cnn_model(image_tensor.unsqueeze(0))

    def predict_batch(self, batch):
        return self.cnn_model(batch).cpu().detach().numpy()
