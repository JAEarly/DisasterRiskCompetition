import torch
import torch.nn.functional as F
from torch import nn

from models import Model


class BasicNN(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NNModel(Model):

    def __init__(self, input_size, state_dict_path=None, eval_mode=False):
        super().__init__("basic_nn")
        self.net = BasicNN(input_size, self.num_classes)
        if state_dict_path is not None:
            self.load(state_dict_path)
        if eval_mode:
            self.net.eval()

    def predict(self, feature_tensor):
        return self.net(feature_tensor)

    def predict_batch(self, feature_batch):
        return self.net(feature_batch)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.net.state_dict(), path)
