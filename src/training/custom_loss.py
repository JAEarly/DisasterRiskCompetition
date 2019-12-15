import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from models import Model


class SmoothedCrossEntropyLoss(Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        ls_pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = torch.mean(torch.sum(-true_dist * ls_pred, dim=self.dim))
        return loss


class PseudoLabelCrossEntropyLoss(Module):

    def __init__(self, pseudo_data_loader: DataLoader, model: Model, alpha: float = 0.1, dim=-1):
        super(PseudoLabelCrossEntropyLoss, self).__init__()
        self.pseudo_data_loader = pseudo_data_loader
        self.model = model
        self.alpha = alpha
        self.dim = dim
        self.pseudo_data_loader_iter = iter(self.pseudo_data_loader)

    def forward(self, pred, target):
        ls_pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            one_hot_targets = torch.zeros_like(pred)
            one_hot_targets.scatter_(1, target.data.unsqueeze(1), 1)
        loss = torch.mean(torch.sum(-one_hot_targets * ls_pred, dim=self.dim))
        pseudo_loss = self.calc_pseudo_loss()
        return loss + self.alpha * pseudo_loss

    def calc_pseudo_loss(self):
        try:
            batch = next(self.pseudo_data_loader_iter)[0]
        except StopIteration:
            self.pseudo_data_loader_iter = iter(self.pseudo_data_loader)
            batch = next(self.pseudo_data_loader_iter)[0]
        pred = self.model.predict_batch(batch)
        ls_pred = pred.log_softmax(dim=self.dim)
        target = torch.zeros_like(pred)
        target.scatter_(1, pred.argmax(dim=-1).unsqueeze(1), 1)
        loss = torch.mean(torch.sum(-target * ls_pred, dim=self.dim))
        return loss

