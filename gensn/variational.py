import numpy as np
import torch
from torch import nn

from .distributions import Joint
from .transforms.surjective import StepQuantizer
from .utils import squeeze_tuple, turn_to_tuple


def ELBO_joint(joint, posterior, *obs, n_samples=1):
    # Joint = p(z, x), Posterior = p(z|x)
    z_samples = posterior.rsample((n_samples,), cond=obs)
    # TODO: take care of case where KL is known for the posterior
    elbo = -posterior(*turn_to_tuple(z_samples), cond=obs)
    elbo += joint(*turn_to_tuple(z_samples), *obs)
    return elbo.mean(dim=0)  # average over samples


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
        self.n_samples = n_samples

    @property
    def n_rvs(self):
        return self.joint.n_rvs - self.posterior.n_rvs

    def forward(self, *obs, cond=None, n_samples=None):
        return self.elbo(*obs, cond=cond, n_samples=n_samples)

    def elbo(self, *obs, cond=None, n_samples=None):
        # TODO: deal with conditioning correctly
        if n_samples is None:
            n_samples = self.n_samples
        return ELBO_joint(
            self.joint,
            self.posterior,
            *obs,
            n_samples=n_samples,
        )

    def log_prob(self, *obs):
        # TODO: let this be implemented as an "approximation" with ELBO
        # but with ample warnings
        pass

    def sample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.joint.sample(sample_shape=sample_shape, cond=cond)
        return squeeze_tuple(samples[-self.n_rvs :])

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.joint.rsample(sample_shape=sample_shape, cond=cond)
        return squeeze_tuple(samples[-self.n_rvs :])


# TODO: this is really no different from simple variational marginal
# Only difference is that conditional distribution is deterministic
# Consider rewriting this as simple variational or as SurVAEFlow
# TODO: this only works in the case where n_rvs = 1
# this is because of the way Quantizer works
class VariationalDequantizedDistribution(nn.Module):
    def __init__(self, prior, dequantizer, quantizer=None, n_samples=1):
        super().__init__()
        self.n_samples = n_samples
        self.prior = prior
        self.dequantizer = dequantizer

        if quantizer is None:
            # default to UnitQuantizer
            quantizer = StepQuantizer()
        self.quantizer = quantizer

    @property
    def n_rvs(self):
        return self.prior.n_rvs

    def forward(self, *obs, cond=None, n_samples=None):
        return self.elbo(*obs, cond=cond, n_samples=n_samples)

    def elbo(self, *obs, cond=None, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        z_samples = self.dequantizer.rsample((n_samples,), cond=obs)
        elbo = -self.dequantizer(*turn_to_tuple(z_samples), cond=obs)
        # TODO: rewrite this so that quantizer can be used as is for joint & elbo
        elbo += self.prior(*turn_to_tuple(z_samples))
        return elbo.mean(dim=0)  # average over samples

    def factorized_elbo(self, *obs, cond=None, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        z_samples = self.dequantizer.rsample((n_samples,), cond=obs)
        log_prob_posterior = self.dequantizer.factorized_log_prob(
            *turn_to_tuple(z_samples), cond=obs
        )
        log_prob_prior = self.prior.factorized_log_prob(*turn_to_tuple(z_samples))
        elbo = log_prob_prior - log_prob_posterior
        return elbo.mean(dim=0)  # average over samples

    def iw_bound(self, *obs, cond=None, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        z_samples = self.dequantizer.rsample((n_samples,), cond=obs)
        log_prob_posterior = self.dequantizer(*turn_to_tuple(z_samples), cond=obs)
        log_prob_prior = self.prior(*turn_to_tuple(z_samples))
        return torch.logsumexp(log_prob_prior - log_prob_posterior, dim=0) - np.log(
            n_samples
        )  # average over samples

    def factorized_iw_bound(self, *obs, cond=None, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        z_samples = self.dequantizer.rsample((n_samples,), cond=obs)
        log_prob_posterior = self.dequantizer.factorized_log_prob(
            *turn_to_tuple(z_samples), cond=obs
        )
        log_prob_prior = self.prior.factorized_log_prob(*turn_to_tuple(z_samples))
        return torch.logsumexp(log_prob_prior - log_prob_posterior, dim=0) - np.log(
            n_samples
        )  # average over samples

    def log_prob(self, *obs):
        # TODO: let this be implemented as an "approximation" with ELBO
        # but with ample warnings
        pass

    def factorized_log_prob(self, *obs):
        pass

    def sample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.prior.sample(sample_shape=sample_shape, cond=cond)
        return self.quantizer(samples)

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        samples = self.prior.rsample(sample_shape=sample_shape, cond=cond)
        return self.quantizer(samples)
