import torch
from torch import nn

from .utils import turn_to_tuple


class FlowDistribution(nn.Module):
    def __init__(self, base_distribution, transform):
        super().__init__()
        self.base_distribution = base_distribution
        self.transform = transform

    @property
    def n_rvs(self):
        return self.base_distribution.n_rvs

    def forward(self, *obs, cond=None):
        return self.log_prob(*obs, cond=cond)

    def log_prob(self, *obs, cond=None):
        x, logL = self.transform(*obs, cond=cond)
        return self.base_distribution.log_prob(*turn_to_tuple(x), cond=cond) + logL

    def factorized_log_prob(self, *obs, cond=None):
        x, logL = self.transform.factorized_forward(*obs, cond=cond)
        return (
            self.base_distribution.factorized_log_prob(*turn_to_tuple(x), cond=cond)
            + logL
        )

    def sample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.base_distribution.sample(sample_shape=sample_shape, cond=cond)
        y, _ = self.transform.inverse(samples, cond=cond)
        return y

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.base_distribution.rsample(sample_shape=sample_shape, cond=cond)
        y, _ = self.transform.inverse(samples, cond=cond)
        return y


class SurVAEFlowDistribution(nn.Module):
    pass
