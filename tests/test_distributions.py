import torch
from torch import nn
from gensn.distributions import TrainableDistributionAdapter
from torch.distributions import Normal
import numpy as np


class TestDistributionAdapterCreation:
    loc = 2.0
    scale = 1.0

    def test_creation_with_values(self):
        td = TrainableDistributionAdapter(Normal, loc=2.0, scale=1.0)
        # here just making sure it runs
        value = td.log_prob(torch.Tensor([1.0]))
        target = np.log(
            np.exp(-((1.0 - self.loc) ** 2) / 2 / self.scale**2)
            / np.sqrt(2 * np.pi * self.scale**2)
        )
        assert np.abs(value - target) < 0.001

    def test_creation_with_parameters(self):
        td = TrainableDistributionAdapter(
            Normal,
            loc=nn.Parameter(torch.Tensor([2.0])),
            scale=nn.Parameter(torch.Tensor([1.0])),
        )

    def test_creation_with_modules(self):
        td = TrainableDistributionAdapter(Normal, loc=nn.Linear(1, 1), scale=3.0)
