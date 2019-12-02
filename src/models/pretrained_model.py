"""Conv neural network solution."""

import torch

from models import Model


class PretrainedNNModel(Model):
    """Base model that uses a pretrained cnn."""

    def __init__(
        self,
        pretrained_net_class,
        final_layer_alteration,
        state_dict_path=None,
        eval_mode=False,
    ):
        super().__init__(str(pretrained_net_class.__name__).lower(), True)
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
