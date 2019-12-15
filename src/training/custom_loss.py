from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn import Module
import torch


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


# class SmoothedCrossEntropyLoss(CrossEntropyLoss):
#
#     def __init__(self, smoothing, weight=None, size_average=None, ignore_index=-100, reduce=None,
#                  reduction='mean'):
#         super(SmoothedCrossEntropyLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
#         self.smoothing = smoothing
#
#     def forward(self, input, target):
#         print("")
#         print(target)
#         smoothed_target = []
#         for t in target:
#             smoothed_t = torch.empty(5)
#             smoothed_t[t] = 1
#             smoothed_t = smoothed_t * (1 - self.smoothing) + self.smoothing / len(smoothed_t)
#             smoothed_target.append(smoothed_t)
#         smoothed_target = torch.stack(smoothed_target)
#         smoothed_target = smoothed_target.to(target.device)
#         print(smoothed_target)
#         return super().forward(input, smoothed_target)
