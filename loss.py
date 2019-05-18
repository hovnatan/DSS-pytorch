import torch.nn.functional as F
from torch import nn


class Loss(nn.Module):
    '''
    loss function: seven probability map --- 6 scale + 1 fuse
    '''

    def __init__(self, weight=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        super(Loss, self).__init__()
        self.weight = weight

    def forward(self, x_list, label):
        losses = []
        for i, x in enumerate(x_list):
            losses.append(self.weight[i] * F.binary_cross_entropy(x, label))
        return sum(losses)
