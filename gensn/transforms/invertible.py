import torch
from torch import nn
import torch.nn.functional as F
from warnings import warn
import math
from ..utils import invoke_with_cond


class SequentialTransform(nn.Module):
    """Defines a transform by combining one or more transforms in sequence."""

    def __init__(self, *transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    # TODO: check the content of nn.Module __call__ and create something similar for inverse
    def forward(self, x, cond=None):
        logL = 0
        for t in self.transforms:
            x, logDet = t.forward(x, cond=cond)
            logL += logDet
        return x, logL

    def inverse(self, y, cond=None):
        logL = 0
        for t in self.transforms[::-1]:
            y, logDet = t.inverse(y, cond=cond)
            logL += logDet
        return y, logL


class InverseTransform(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x, cond=None):
        return self.transform.inverse(x, cond=cond)

    def inverse(self, y, cond=None):
        return self.transform.forward(y, cond=cond)


class ConditionalShift(nn.Module):
    def __init__(self, conditional_shift):
        super().__init__()
        self.conditional_shift = conditional_shift

    def forward(self, x, cond=None):
        x = x + invoke_with_cond(self.conditional_shift, cond=cond)
        return x, 0

    def inverse(self, x, cond=None):
        x = x - invoke_with_cond(self.conditional_shift, cond=cond)
        return x, 0


# # conceptual template
# class InvertibleTransform(nn.Module):
#     def forward(self, x, logL=0):
#         return y, logL + log_det_f_prime

#     def inverse(self, y, logL=0):
#         return x, logL - log_det_f_prime


class MarginalTransform(nn.Module):
    """Defines marginal transform template"""

    def __init__(self, dim=-1):
        """Initialize the marginal transform. Specify the dimension to be
        collapsed over when computing the log determinant.

        Args:
            dim (int or tuple, optional): Dimension(s) to collapse over when computing the log determinant. Defaults to -1 (the last dimension).
        """
        super().__init__()
        self.dim = dim

    def marginal_forward(self, x):
        pass

    def marginal_inverse(self, y):
        pass

    def get_log_det(self, x):
        pass

    def forward(self, x, cond=None):
        # TODO: deal with passing the cond through
        return self.marginal_forward(x), self.get_log_det(x).sum(dim=self.dim)

    def inverse(self, y, cond=None):
        x = self.marginal_inverse(y)
        return x, -self.get_log_det(x)


class IndependentAffine(MarginalTransform):
    def __init__(self, input_dim=1, dim=-1):
        super().__init__(dim=dim)
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.empty(input_dim))
        self.bias = nn.Parameter(torch.empty(input_dim))

    def get_log_det(self, x):
        return torch.log(
            abs(self.weight) + torch.finfo(self.weight.dtype).tiny
        ) * torch.ones_like(x)

    def marginal_forward(self, x):
        return x * self.weight + self.bias

    def marginal_inverse(self, y):
        return (y - self.bias) / self.weight


class ELU(MarginalTransform):
    def __init__(self, alpha=1.0, dim=-1):
        super().__init__(dim=dim)
        self.alpha = alpha

    def get_log_det(self, x):
        return torch.where(x > 0, torch.zeros(1).to(x.device), x + math.log(self.alpha))

    def marginal_forward(self, x, cond=None):
        return F.elu(x, self.alpha)

    def marginal_inverse(self, y, cond=None):
        # TODO: check if use of finfo is meaningful
        finfo = torch.finfo(y.dtype)
        return torch.where(
            y > 0, y, torch.log((y / self.alpha + 1).clamp(min=finfo.tiny))
        )


class OffsetELU(nn.Module):
    def __init__(self, alpha=1.0, offset=0.0, dim=-1):
        super().__init__()
        self.elu = ELU(alpha, dim=dim)
        self.offset = offset

    def forward(self, x, cond=None):
        y, logL = self.elu(x, cond=cond)
        return y + self.offset, logL

    def inverse(self, y, cond=None):
        return self.elu.inverse(y - self.offset, cond=cond)


class ELUplus1(OffsetELU):
    def __init__(self, alpha=1.0, dim=-1):
        super().__init__(alpha=alpha, offset=1.0, dim=dim)


class Softplus(MarginalTransform):
    def _get_marginal_log_det(self, x):
        # TODO: get the implementation for softplus
        return -F.softplus(-x)

    def marginal_forward(self, x):
        return F.softplus(x)

    def marginal_inverse(self, y):
        # TODO: consider providing inverse_softplus
        return (-y).expm1().neg().clamp(min=torch.finfo(y.dtype).tiny).log() + y


class Exp(MarginalTransform):
    def get_log_det(self, x):
        return x

    def marginal_forward(self, x):
        return x.exp()

    def marginal_inverse(self, y):
        return y.clamp(min=torch.finfo(y.dtype).tiny).log()


class Tanh(MarginalTransform):
    def get_log_det(self, x):
        # using numerically stable formula from TF implementation:
        # https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

    def marginal_forward(self, x):
        return torch.tanh(x)

    # switchable safeguarding?
    # publicized?
    # this is silently failing for atanh out of bounds inputs
    def marginal_inverse(self, y):
        eps = torch.finfo(y.dtype).eps
        return y.clamp(min=-1 + eps, max=1 - eps).atanh()


class Sigmoid(MarginalTransform):
    # TODO: implement scaling and also top & bottom offset
    def get_log_det(self, x):
        return -F.softplus(-x) - F.softplus(x)

    def marginal_forward(self, x):
        return torch.sigmoid(x)

    def marginal_inverse(self, y):
        finfo = torch.finfo(y.dtype)
        y = torch.clamp(y, min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()


class Log(MarginalTransform):
    def get_log_det(self, x):
        return -torch.log(abs(x).clamp(min=torch.finfo(x.dtype).tiny))

    def marginal_forward(self, x, cond=None):
        return x.clamp(min=torch.finfo(x.dtype).tiny).log()

    def marginal_inverse(self, y, cond=None):
        return y.exp()


class Pow(MarginalTransform):
    def get_log_det(self, x):
        # TODO: deal with number/tensor conversion better here
        # currently using torch.zeros to ensure sum is a tensor
        return torch.log(
            torch.zeros([]) + abs(self.exponent) + torch.finfo(x.dtype).tiny
        ) + (self.exponent - 1) * torch.log(abs(x) + torch.finfo(x.dtype).tiny)

    def __init__(self, exponent, dim=-1):
        super().__init__(dim=dim)
        self.exponent = exponent

    def marginal_forward(self, x):
        return x.pow(self.exponent)

    def marginal_inverse(self, y):
        return y.pow(1 / (self.exponent + torch.finfo(y.dtype).tiny))


class Sqrt(MarginalTransform):
    def get_log_det(self, x):
        # TODO: replace with torch.log
        return -math.log(2.0) - 0.5 * torch.log(abs(x) + torch.finfo(x.dtype).tiny)

    def marginal_forward(self, x):
        # TODO: Evaluate if this clamping is a good idea
        return x.clamp(min=0).sqrt()

    def marginal_inverse(self, z):
        return z.pow(2)
