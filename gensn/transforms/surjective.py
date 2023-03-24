# collection of quantizers and dequantizers

import torch
from torch import nn


# TODO: this only works for the case of n_rvs = 1
class StepQuantizer(nn.Module):
    # TODO: consider calling it center instead of shift
    def __init__(self, step=1, shift=0):
        super().__init__()
        self.step = step
        self.shift = shift

    def forward(self, z):
        return (
            torch.floor((z - self.shift) / self.step).to(z.dtype) * self.step
            + self.shift
        )
