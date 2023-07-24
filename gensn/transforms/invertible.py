import math
from warnings import warn

import torch
import torch.nn.functional as F
from torch import nn

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

    def factorized_forward(self, x, cond=None):
        logL = 0
        for t in self.transforms:
            x, logDet = t.factorized_forward(x, cond=cond)
            logL += logDet
        return x, logL

    def factorized_inverse(self, y, cond=None):
        logL = 0
        for t in self.transforms[::-1]:
            y, logDet = t.factorized_inverse(y, cond=cond)
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

    def factorized_forward(self, x, cond=None):
        return self.transform.factorized_inverse(x, cond=cond)

    def factorized_inverse(self, y, cond=None):
        return self.transform.factorized_forward(y, cond=cond)


class ConditionalShift(nn.Module):
    def __init__(self, conditional_shift):
        super().__init__()
        self.conditional_shift = conditional_shift

    def forward(self, x, cond=None):
        x = x + invoke_with_cond(self.conditional_shift, cond=cond)
        return x, 0

    def factorized_forward(self, x, cond=None):
        return self(x, cond=cond)

    def inverse(self, x, cond=None):
        x = x - invoke_with_cond(self.conditional_shift, cond=cond)
        return x, 0

    def factorized_inverse(self, x, cond=None):
        return self.inverse(x, cond=cond)


# # conceptual template
# class InvertibleTransform(nn.Module):
#     def forward(self, x, logL=0):
#         return y, logL + log_det_f_prime

#     def inverse(self, y, logL=0):
#         return x, logL - log_det_f_prime


class FactorizedTransform(nn.Module):
    """Defines factorized transform template"""

    def __init__(self, dim=-1):
        """Initialize the factorized transform. Specify the dimension to be
        collapsed over when computing the log determinant.

        Args:
            dim (int or tuple, optional): Dimension(s) to collapse over when computing the log determinant. Defaults to -1 (the last dimension).
        """
        super().__init__()
        self.dim = dim

    def factorized_transform(self, x, cond=None):
        pass

    def factorized_inverse_transform(self, y, cond=None):
        pass

    def get_log_det(self, x, cond=None):
        pass

    def factorized_forward(self, x, cond=None):
        return self.factorized_transform(x, cond=cond), self.get_log_det(x, cond=cond)

    def forward(self, x, cond=None):
        y, log_det = self.factorized_forward(x, cond=cond)
        return y, log_det.sum(dim=self.dim)

    def factorized_inverse(self, y, cond=None):
        x = self.factorized_inverse_transform(y, cond=cond)
        return x, -self.get_log_det(x, cond=cond)

    def inverse(self, y, cond=None):
        x, log_det = self.factorized_inverse(y, cond=cond)
        return x, log_det.sum(dim=self.dim)


class IndependentAffine(FactorizedTransform):
    def __init__(self, input_dim=1, dim=-1):
        super().__init__(dim=dim)
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.empty(input_dim))
        self.bias = nn.Parameter(torch.empty(input_dim))

    def get_log_det(self, x, cond=None):
        return torch.log(
            abs(self.weight) + torch.finfo(self.weight.dtype).tiny
        ) * torch.ones_like(x)

    def factorized_transform(self, x, cond=None):
        return x * self.weight + self.bias

    def factorized_inverse_transform(self, y, cond=None):
        return (y - self.bias) / (self.weight + torch.finfo(self.weight.dtype).tiny)


class ELU(FactorizedTransform):
    def __init__(self, alpha=1.0, dim=-1):
        super().__init__(dim=dim)
        self.alpha = alpha

    def get_log_det(self, x, cond=None):
        return torch.where(x > 0, torch.zeros(1).to(x.device), x + math.log(self.alpha))

    def factorized_transform(self, x, cond=None):
        return F.elu(x, self.alpha)

    def factorized_inverse_transform(self, y, cond=None):
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


class Softplus(FactorizedTransform):
    def get_log_det(self, x, cond=None):
        # TODO: get the implementation for softplus
        return -F.softplus(-x)

    def factorized_transform(self, x, cond=None):
        return F.softplus(x)

    def factorized_inverse_transform(self, y, cond=None):
        # TODO: consider providing inverse_softplus
        return (-y).expm1().neg().clamp(min=torch.finfo(y.dtype).tiny).log() + y


class Exp(FactorizedTransform):
    def get_log_det(self, x, cond=None):
        return x

    def factorized_transform(self, x, cond=None):
        return x.exp()

    def factorized_inverse_transform(self, y, cond=None):
        return y.clamp(min=torch.finfo(y.dtype).tiny).log()


class Tanh(FactorizedTransform):
    def get_log_det(self, x, cond=None):
        # using numerically stable formula from TF implementation:
        # https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

    def factorized_transform(self, x, cond=None):
        return torch.tanh(x)

    # switchable safeguarding?
    # publicized?
    # this is silently failing for atanh out of bounds inputs
    def factorized_inverse_transform(self, y, cond=None):
        eps = torch.finfo(y.dtype).eps
        return y.clamp(min=-1 + eps, max=1 - eps).atanh()


class Sigmoid(FactorizedTransform):
    # TODO: implement scaling and also top & bottom offset
    def get_log_det(self, x, cond=None):
        return -F.softplus(-x) - F.softplus(x)

    def factorized_transform(self, x, cond=None):
        return torch.sigmoid(x)

    def factorized_inverse_transform(self, y, cond=None):
        finfo = torch.finfo(y.dtype)
        y = torch.clamp(y, min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()


class Log(FactorizedTransform):
    def get_log_det(self, x, cond=None):
        return -torch.log(abs(x).clamp(min=torch.finfo(x.dtype).tiny))

    def factorized_forward(self, x, cond=None):
        return x.clamp(min=torch.finfo(x.dtype).tiny).log()

    def factorized_inverse_transform(self, y, cond=None):
        return y.exp()


class Pow(FactorizedTransform):
    def __init__(self, exponent, dim=-1):
        super().__init__(dim=dim)
        self.exponent = exponent

    def get_log_det(self, x, cond=None):
        # TODO: deal with number/tensor conversion better here
        # currently using torch.zeros to ensure sum is a tensor
        return torch.log(
            torch.zeros([]) + abs(self.exponent) + torch.finfo(x.dtype).tiny
        ) + (self.exponent - 1) * torch.log(abs(x) + torch.finfo(x.dtype).tiny)

    def factorized_transform(self, x, cond=None):
        return x.pow(self.exponent)

    def factorized_inverse_transform(self, y, cond=None):
        return y.pow(1 / (self.exponent + torch.finfo(y.dtype).tiny))


class Sqrt(FactorizedTransform):
    def get_log_det(self, x, cond=None):
        # TODO: replace with torch.log
        return -math.log(2.0) - 0.5 * torch.log(abs(x) + torch.finfo(x.dtype).tiny)

    def factorized_transform(self, x, cond=None):
        # TODO: Evaluate if this clamping is a good idea
        return x.clamp(min=0).sqrt()

    def factorized_inverse_transform(self, z, cond=None):
        return z.pow(2)
