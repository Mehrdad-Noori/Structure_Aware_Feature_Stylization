import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        batch_size = inputs.size(dim=0)

        inputs = inputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        dice_loss = 0
        bce = 0

        for i in range(batch_size):
            input = inputs[i]
            target = targets[i]
            intersection = (input * target).sum()
            dice_loss += 1 - ((2. * intersection + self.smooth) / (input.sum() + target.sum() + self.smooth))
            bce += F.binary_cross_entropy(input, target, reduction="mean")

        dice_loss /= batch_size
        bce /= batch_size

        loss = bce + dice_loss

        return loss
