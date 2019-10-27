from abc import abstractmethod, ABC

from .model import Model


class TransferModel(Model, ABC):

    def __init__(self, name, state_dict_path=None, eval_mode=False):
        super().__init__(name)
        self.transfer_model = self.setup_transfer_model(state_dict_path, eval_mode)

    @abstractmethod
    def setup_transfer_model(self, state_dict_path, eval_mode):
        pass

    def predict(self, image_tensor):
        return self.transfer_model(image_tensor.unsqueeze(0)).cpu().detach().numpy()

    def predict_batch(self, batch):
        return self.transfer_model(batch).cpu().detach().numpy()
