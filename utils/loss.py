import numpy as np
from torch import nn


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positive, negative):
        distance = positive - negative
        loss = np.sum(np.log(nn.Sigmoid(distance)), axis=0, keepdim=True)
        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positive, negative, margin=1):
        distance = positive - negative
        loss = np.sum(np.maximum(margin - distance, 0))
        return loss
