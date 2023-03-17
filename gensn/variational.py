import torch
from torch import nn
from .utils import turn_to_tuple
from .distributions import Joint


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
