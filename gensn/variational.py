import torch
from torch import nn
from .utils import turn_to_tuple
from .distributions import Joint
from .transforms.surjective import StepQuantizer


def ELBO_joint(joint, posterior, *obs, n_samples=1):
    # Joint = p(z, x), Posterior = p(z|x)
    z_samples = posterior.rsample((n_samples,), cond=obs)
    # take care of case where KL is known for the posterior
    elbo = -posterior(*turn_to_tuple(z_samples), cond=obs)
    elbo += joint(*turn_to_tuple(z_samples), *obs)
    return elbo


def ELBO_parts(prior, conditional, posterior, *obs, n_samples=1):
    # create a joint
    joint = Joint(prior, conditional)
    return ELBO_joint(joint, posterior, *obs, n_samples=n_samples)


class ELBOMarginal(nn.Module):
    approximate_log_prob = True

    def __init__(self, joint, posterior, n_samples=1):
        super().__init__()
        self.joint = joint
        self.posterior = posterior
        # infer how many variables are in observations
        self.n_rvs = joint.n_rvs - posterior.n_rvs
        self.n_samples = n_samples

    def forward(self, *obs, cond=None):
        return self.elbo(*obs, cond=cond)

    def elbo(self, *obs, cond=None):
        # TODO: deal with conditioning correctly
        return ELBO_joint(self.joint, self.posterior, *obs, n_samples=self.n_samples)

    def log_prob(self, *obs):
        # TODO: let this be implemented as an "approximation" with ELBO
        # but with ample warnings
        pass

    def sample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.joint.sample(sample_shape=sample_shape, cond=cond)
        return samples[-self.n_rvs :]

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.joint.rsample(sample_shape=sample_shape, cond=cond)
        return samples[-self.n_rvs :]


# TODO: this is really no different from simple variational marginal
# Only difference is that conditional distribution is deterministic
# Consider rewriting this as simple variational or as SurVAEFlow
class VariationalDequantizedDistribution(nn.Module):
    def __init__(
        self, dequantized_distribution, dequantizer, quantizer=None, n_samples=1
    ):
        super().__init__()
        self.n_samples = n_samples
        self.dequantized_distribution = dequantized_distribution
        self.dequantizer = dequantizer

        if quantizer is None:
            # default to UnitQuantizer
            quantizer = StepQuantizer()
        self.quantizer = quantizer

    def forward(self, *obs, cond=None):
        return self.elbo(*obs, cond=cond)

    def elbo(self, *obs, cond=None):
        z_samples = self.dequantizer.rsample((self.n_samples,), cond=obs)
        elbo = -self.dequantizer(*turn_to_tuple(z_samples), cond=obs)
        # TODO: rewrite this so that quantizer can be used as is for joint & elbo
        elbo += self.dequantized_distribution(*turn_to_tuple(z_samples))
        return elbo

    def log_prob(self, *obs):
        # TODO: let this be implemented as an "approximation" with ELBO
        # but with ample warnings
        pass

    def sample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.dequantized_distribution.sample(
            sample_shape=sample_shape, cond=cond
        )
        return self.quantizer(samples)

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.dequantized_distribution.rsample(
            sample_shape=sample_shape, cond=cond
        )
        return self.quantizer(samples)
