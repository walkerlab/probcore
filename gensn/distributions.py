from abc import ABC, abstractmethod

import torch
import torch.distributions as D
from torch import nn

from .utils import invoke_with_cond, make_args, register_to_module, turn_to_tuple


# come up with a better name
# here we assume that conditioning of the joint distribution can occur
# by conditioning the prior
class Joint(nn.Module):
    def __init__(self, prior, conditional):
        super().__init__()
        self.prior = prior
        self.conditional = conditional

    @property
    def n_rvs(self):
        return self.prior.n_rvs + self.conditional.n_rvs

    def log_prob(self, *obs, cond=None):
        x, y = obs[: self.prior.n_rvs], obs[self.prior.n_rvs :]
        return self.prior(*x, cond=cond) + self.conditional(*y, cond=x)

    def factorized_log_prob(self, *obs, cond=None):
        x, y = obs[: self.prior.n_rvs], obs[self.prior.n_rvs :]
        return self.prior.factorized_log_prob(
            *x, cond=cond
        ) + self.conditional.factorized_log_prob(*y, cond=x)

    def forward(self, *obs, cond=None):
        return self.log_prob(*obs, cond=cond)

    def sample(self, sample_shape=torch.Size([]), cond=None):
        x_samples = self.prior.sample(sample_shape=sample_shape, cond=cond)
        y_samples = self.conditional.sample(cond=x_samples)
        return turn_to_tuple(x_samples) + turn_to_tuple(y_samples)

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        x_samples = self.prior.rsample(sample_shape=sample_shape, cond=cond)
        y_samples = self.conditional.rsample(cond=x_samples)
        return turn_to_tuple(x_samples) + turn_to_tuple(y_samples)


class TrainableDistribution(nn.Module, ABC):
    """
    Here we are providing the proper abstract base class for the
    TrainableDistribution. However, at the moment, this is not used
    anywhere and only serving to indicate what methods must be implemented
    for any object to be considered a proper, trainable distribution.
    """

    @property
    @abstractmethod
    def n_rvs(self):
        """
        Defines the number of random variables that this distribution is
        defined over.
        """
        ...

    @abstractmethod
    def log_prob(self, *obs, cond=None):
        ...

    def forward(self, *obs, cond=None):
        return self.log_prob(*obs, cond=cond)

    @abstractmethod
    def sample(self, sample_shape=torch.Size([]), cond=None):
        ...

    @abstractmethod
    def rsample(self, sample_shape=torch.Size([]), cond=None):
        ...


class TrainableDistributionAdapter(nn.Module):
    n_rvs = 1

    def __init__(self, distribution_class, *dist_args, _parameters=None, **dist_kwargs):
        super().__init__()
        self.distribution_class = distribution_class
        self.param_counts = len(dist_args)
        self.param_keys = list(dist_kwargs.keys())

        for pos, val in enumerate(dist_args):
            # setattr(self, f'_arg{pos}', val)
            register_to_module(self, f"_arg{pos}", val)
        for key, val in dist_kwargs.items():
            register_to_module(self, key, val)

        # The alternative interface to allow for a single module to
        # flexibly output multiple parameters for the distribution
        # at the moment, the module is expected to output a dictionary
        # of parameters
        if _parameters is not None:
            self.parameter_generator = _parameters

    def distribution(self, cond=None):
        cond = turn_to_tuple(cond)

        dist_args = tuple(
            invoke_with_cond(getattr(self, f"_arg{pos}"), cond=cond)
            for pos in range(self.param_counts)
        )
        dist_kwargs = {
            k: invoke_with_cond(getattr(self, k), cond=cond) for k in self.param_keys
        }

        # TODO: consider flipping the order of this with
        # init specified parameters
        if hasattr(self, "parameter_generator"):
            dist_args, dist_kwargs = make_args(
                self.parameter_generator(*cond), *dist_args, **dist_kwargs
            )

        return self.distribution_class(*dist_args, **dist_kwargs)

    def log_prob(self, *obs, cond=None):
        return self.distribution(cond=cond).log_prob(*obs)

    def forward(self, *obs, cond=None):
        return self.log_prob(*obs, cond=cond)

    def sample(self, sample_shape=torch.Size([]), cond=None):
        return self.distribution(cond=cond).sample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        return self.distribution(cond=cond).rsample(sample_shape=sample_shape)

    # overwrite extra_repr to include the distribution class
    # TODO: consider adding the parameters as well
    def extra_repr(self):
        repr = f"distribution_class={self.distribution_class!r}"

        if self.param_counts > 0:
            repr += ", " + ", ".join(
                f"{getattr(self, f'_arg{pos}')!r}" for pos in range(self.param_counts)
            )

        if len(self.param_keys) > 0:
            repr += ", " + ", ".join(
                f"{k}={getattr(self, k)!r}" for k in self.param_keys
            )

        if hasattr(self, "parameter_genrator"):
            repr += ", " + f"_parameters={self.parameter_generator!r}"

        return repr


class IndependentTrainableDistributionAdapter(TrainableDistributionAdapter):
    def __init__(
        self,
        distribution_class,
        *dist_args,
        _parameters=None,
        event_dims=1,
        **dist_kwargs,
    ):
        super().__init__(
            distribution_class, *dist_args, _parameters=_parameters, **dist_kwargs
        )
        self.event_dims = event_dims

    def distribution(self, cond=None):
        return D.Independent(super().distribution(cond=cond), self.event_dims)

    def factorized_distribution(self, cond=None):
        return super().distribution(cond=cond)

    def factorized_log_prob(self, *obs, cond=None):
        return self.factorized_distribution(cond=cond).log_prob(*obs)

    def extra_repr(self):
        return super().extra_repr() + f", event_dims={self.event_dims}"


class WrappedTrainableDistribution(nn.Module):
    def __init__(self, trainable_distribution=None):
        super().__init__()
        self.trainable_distribution = trainable_distribution

    @property
    def n_rvs(self):
        return self.trainable_distribution.n_rvs

    def forward(self, *obs, cond=None):
        return self.trainable_distribution(*obs, cond=cond)

    def log_prob(self, *obs, cond=None):
        return self.trainable_distribution.log_prob(*obs, cond=cond)

    def factorized_log_prob(self, *obs, cond=None):
        return self.trainable_distribution.factorized_log_prob(*obs, cond=cond)

    def sample(self, sample_shape=torch.Size([]), cond=None):
        return self.trainable_distribution.sample(sample_shape=sample_shape, cond=cond)

    def rsample(self, sample_shape=torch.Size([]), cond=None):
        return self.trainable_distribution.rsample(sample_shape=sample_shape, cond=cond)


class IndependentNormal(WrappedTrainableDistribution):
    """
    A trainable distribution that wraps a D.Independent(D.Normal) distribution
    """

    def __init__(self, loc=None, scale=None, _parameters=None, event_dims=1):
        """
        Args:
            loc : torch.Tensor or nn.Parameter or None
                The mean of the normal distribution. If None, _parameters must be provided
            scale : torch.Tensor or nn.Parameter or None
                The standard deviation of the normal distribution. If None, _parameters must be provided
            _parameters : callable or None
                A function that takes in the conditioning variable and returns a dictionary of parameters
                for the normal distribution. If None, loc and scale must be provided
            event_dims : int
                The number of dimensions to be considered as the event dimensions
        """
        super().__init__()
        if (loc is None or scale is None) and _parameters is None:
            raise ValueError(
                "If either loc or scale is unspecificed, _parameters must be provided"
            )
        kwargs = {}
        if loc is not None:
            kwargs["loc"] = loc
        if scale is not None:
            kwargs["scale"] = scale
        self.trainable_distribution = IndependentTrainableDistributionAdapter(
            D.Normal,
            event_dims=event_dims,
            **kwargs,
            _parameters=_parameters,
        )


class IndependentGamma(WrappedTrainableDistribution):
    """
    A trainable distribution that wraps a D.Independent(D.Gamma) distribution
    """

    def __init__(self, concentration=None, rate=None, _parameters=None, event_dims=1):
        """
        Args:
            concentration : torch.Tensor or nn.Parameter or None
                The concentration parameter of the gamma distribution. If None, _parameters must be provided
            rate : torch.Tensor or nn.Parameter or None
                The rate parameter of the gamma distribution. If None, _parameters must be provided
            _parameters : callable or None
                A function that takes in the conditioning variable and returns a dictionary of parameters
                for the gamma distribution. If None, concentration and rate must be provided
            event_dims : int
                The number of dimensions to be considered as the event dimensions
        """
        super().__init__()
        if (concentration is None or rate is None) and _parameters is None:
            raise ValueError(
                "If either concentration or rate is unspecificed, _parameters must be provided"
            )
        kwargs = {}
        if concentration is not None:
            kwargs["concentration"] = concentration
        if rate is not None:
            kwargs["rate"] = rate
        self.trainable_distribution = IndependentTrainableDistributionAdapter(
            D.Gamma,
            event_dims=event_dims,
            **kwargs,
            _parameters=_parameters,
        )


class IndependentLaplace(WrappedTrainableDistribution):
    """
    A trainable distribution that wraps a D.Independent(D.Laplace) distribution
    """

    def __init__(self, loc=None, scale=None, _parameters=None, event_dims=1):
        """
        Args:
            loc : torch.Tensor or nn.Parameter or None
                The mean of the laplace distribution. If None, _parameters must be provided
            scale : torch.Tensor or nn.Parameter or None
                The scale parameter of the laplace distribution. If None, _parameters must be provided
            _parameters : callable or None
                A function that takes in the conditioning variable and returns a dictionary of parameters
                for the laplace distribution. If None, loc and scale must be provided
            event_dims : int
                The number of dimensions to be considered as the event dimensions
        """
        super().__init__()
        if (loc is None or scale is None) and _parameters is None:
            raise ValueError(
                "If either loc or scale is unspecificed, _parameters must be provided"
            )
        kwargs = {}
        if loc is not None:
            kwargs["loc"] = loc
        if scale is not None:
            kwargs["scale"] = scale
        self.trainable_distribution = IndependentTrainableDistributionAdapter(
            D.Laplace,
            event_dims=event_dims,
            **kwargs,
            _parameters=_parameters,
        )


class IndependentExponential(WrappedTrainableDistribution):
    """
    A trainable distribution that wraps a D.Independent(D.Exponential) distribution
    """

    def __init__(self, rate=None, _parameters=None, event_dims=1):
        """
        Args:
            rate : torch.Tensor or nn.Parameter or None
                The rate parameter of the exponential distribution. If None, _parameters must be provided
            _parameters : callable or None
                A function that takes in the conditioning variable and returns a dictionary of parameters
                for the exponential distribution. If None, rate must be provided
            event_dims : int
                The number of dimensions to be considered as the event dimensions
        """
        super().__init__()
        if rate is None and _parameters is None:
            raise ValueError("If rate is unspecificed, _parameters must be provided")
        kwargs = {}
        if rate is not None:
            kwargs["rate"] = rate
        self.trainable_distribution = IndependentTrainableDistributionAdapter(
            D.Exponential,
            event_dims=event_dims,
            **kwargs,
            _parameters=_parameters,
        )


class IndependentHalfNormal(WrappedTrainableDistribution):
    """
    A trainable distribution that wraps a D.Independent(D.HalfNormal) distribution
    """

    def __init__(self, scale=None, _parameters=None, event_dims=1):
        """
        Args:
            scale : torch.Tensor or nn.Parameter or None
                The scale parameter of the half normal distribution. If None, _parameters must be provided
            _parameters : callable or None
                A function that takes in the conditioning variable and returns a dictionary of parameters
                for the half normal distribution. If None, scale must be provided
            event_dims : int
                The number of dimensions to be considered as the event dimensions
        """
        super().__init__()
        if scale is None and _parameters is None:
            raise ValueError("If scale is unspecificed, _parameters must be provided")
        kwargs = {}
        if scale is not None:
            kwargs["scale"] = scale
        self.trainable_distribution = IndependentTrainableDistributionAdapter(
            D.HalfNormal,
            event_dims=event_dims,
            **kwargs,
            _parameters=_parameters,
        )


class IndependentLogNormal(WrappedTrainableDistribution):
    """
    A trainable distribution that wraps a D.Independent(D.LogNormal) distribution
    """

    def __init__(self, loc=None, scale=None, _parameters=None, event_dims=1):
        """
        Args:
            loc : torch.Tensor or nn.Parameter or None
                The mean of the log normal distribution. If None, _parameters must be provided
            scale : torch.Tensor or nn.Parameter or None
                The scale parameter of the log normal distribution. If None, _parameters must be provided
            _parameters : callable or None
                A function that takes in the conditioning variable and returns a dictionary of parameters
                for the log normal distribution. If None, loc and scale must be provided
            event_dims : int
                The number of dimensions to be considered as the event dimensions
        """
        super().__init__()
        if (loc is None or scale is None) and _parameters is None:
            raise ValueError(
                "If either loc or scale is unspecificed, _parameters must be provided"
            )
        kwargs = {}
        if loc is not None:
            kwargs["loc"] = loc
        if scale is not None:
            kwargs["scale"] = scale
        self.trainable_distribution = IndependentTrainableDistributionAdapter(
            D.LogNormal,
            event_dims=event_dims,
            **kwargs,
            _parameters=_parameters,
        )


class IndependentPoisson(WrappedTrainableDistribution):
    """
    A trainable distribution that wraps a D.Independent(D.Poisson) distribution
    """

    def __init__(self, rate=None, _parameters=None, event_dims=1):
        """
        Args:
            rate : torch.Tensor or nn.Parameter or None
                The rate parameter of the poisson distribution. If None, _parameters must be provided
            _parameters : callable or None
                A function that takes in the conditioning variable and returns a dictionary of parameters
                for the poisson distribution. If None, rate must be provided
            event_dims : int
                The number of dimensions to be considered as the event dimensions
        """
        super().__init__()
        if rate is None and _parameters is None:
            raise ValueError("If rate is unspecificed, _parameters must be provided")
        kwargs = {}
        if rate is not None:
            kwargs["rate"] = rate
        self.trainable_distribution = IndependentTrainableDistributionAdapter(
            D.Poisson,
            event_dims=event_dims,
            **kwargs,
            _parameters=_parameters,
        )


# class DeltaDistribution(nn.Module):
#     def __init__(self, value):
#         self.value = value

#     def log_prob(self, obs, cond=None):
#         # TODO: write to deal with more than one rvs
#         return torch.log(self.prob(obs, cond=cond))

#     def prob(self, obs, cond=None):
#         # TODO: write to deal with more than one rvs
#         return torch.where(
#             torch.equal(obs, parse_attr(self.value, cond=cond)), 0, 1
#         ).to(obs.device)

#     def sample(self, sample_shape=torch.size([]), cond=None):


# def wrap_with_indep(distribution_class, event_dims=1):
#     """
#     Wrap the construction of the target distribution `distr_class` with
#     D.Independent. The returned function can be used as if it is
#     a constructor for an indepdenent version of the distribution
#     """

#     def indep_distr(*args, **kwargs):
#         return D.Independent(distribution_class(*args, **kwargs), event_dims)

#     return indep_distr
