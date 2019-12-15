import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from models import Model


class SmoothedPseudoCrossEntropyLoss(Module):

    def __init__(self, pseudo_data_loader: DataLoader, model: Model, smoothing=0.0, alpha=0.0, dim=-1):
        super(SmoothedPseudoCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.alpha = alpha
        self.dim = dim
        self.pseudo_data_loader = pseudo_data_loader
        self.model = model
        self.pseudo_data_loader_iter = iter(self.pseudo_data_loader)

    def forward(self, pred, target):
        loss = self._calc_main_loss(pred, target)
        pseudo_loss = self._calc_pseudo_loss()
        return loss + self.alpha * pseudo_loss

    def _calc_main_loss(self, pred, target):
        target = self._smooth(pred, target)
        return self._log_loss(pred, target)

    def _calc_pseudo_loss(self):
        try:
            batch = next(self.pseudo_data_loader_iter)[0]
        except StopIteration:
            self.pseudo_data_loader_iter = iter(self.pseudo_data_loader)
            batch = next(self.pseudo_data_loader_iter)[0]
        pred = self.model.predict_batch(batch)
        target = pred.argmax(dim=-1)
        target = self._smooth(pred, target)
        return self._log_loss(pred, target)

    def _smooth(self, pred, target):
        with torch.no_grad():
            smoothed_dist = torch.zeros_like(pred)
            smoothed_dist.fill_(self.smoothing / (self.model.num_classes - 1))
            smoothed_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return smoothed_dist

    def _log_loss(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        return torch.mean(torch.sum(-target * pred, dim=self.dim))
