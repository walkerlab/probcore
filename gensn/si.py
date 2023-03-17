import torch
from torch import nn
from .distributions import TrainableDistributionAdapter


class ProbabilisticSIModel(nn.Module):
    # TODO: for this to work, the TrainableDistributionAdapter handling of
    # _parameters must be expanded. Namely, it needs to be able to accept:
    # * positional arguments (to be implemented)
    # * dict (already implemented)
    # * dict with positional arguments (to be implemented)
    #
    # To also make this generically useful, it would be helpful to
    # allow for output conversion function to be supplied. This function then
    # should transform outputs of the SI model into format appropriate
    # to serve as _parameters for the TrainableDistributionAdapter
    # It's important that such transformation does NOT warp the output
    # Doing so will distort the probability density!
    def __init__(self, si_model, distribution_class, *dist_args, **dist_kwargs):
        super().__init__()
        self.si_model = si_model
        self.distribution_class = distribution_class
        self.trainable_distribution = TrainableDistributionAdapter(
            distribution_class, *dist_args, _parameters=si_model, **dist_kwargs
        )

    def log_prob(self, *obs, cond=None):
        return self.trainable_distribution.log_prob(*obs, cond=cond)

    def forward(self, *obs, cond=None):
        return self.log_prob(*obs, cond=cond)

    def sample(self, sample_shape=torch.Size([]), cond=None):
        return self.trainable_distribution.sample(sample_shape=sample_shape, cond=cond)

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        return self.trainable_distribution.rsample(sample_shape=sample_shape, cond=cond)
