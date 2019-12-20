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
        num_classes=5,
    ):
        super().__init__(str(pretrained_net_class.__name__).lower(), True, num_classes=num_classes)
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
        return torch.softmax(self.net(feature_tensor.to(self.device)), 1)

    def predict_batch(self, feature_batch):
        return torch.softmax(self.net(feature_batch.to(self.device)), 1)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    @staticmethod
    def create_from_transfer(pretrained_net_class, final_layer_alteration, original_model_path, original_num_classes, new_number_classes):
        model = PretrainedNNModel(pretrained_net_class, final_layer_alteration, state_dict_path=original_model_path, num_classes=original_num_classes)
        final_layer_alteration(model.net, new_number_classes)
        return model
