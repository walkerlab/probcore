# collection of quantizers and dequantizers

import torch
from torch import nn


class StepQuantizer(nn.Module):
    def __init__(self, step=1):
        super().__init__()
        self.step = step

    def forward(self, z):
        return torch.floor(z / self.step).to(z.dtype) * self.step
