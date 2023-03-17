import torch
from torch import nn


class TransformedParameter(nn.Module):
    def __init__(self, tensor, transform_fn=None):
        super().__init__()
        self.parameter = nn.Parameter(tensor)
        if transform_fn is None:
            transform_fn = lambda x: x
        self.transform_fn = transform_fn

    @property
    def value(self):
        return self()

    def forward(self, *args):
        return self.transform_fn(self.parameter)


class Covariance(nn.Module):
    def __init__(self, n_dims, rank=None, eps=1e-16):
        super().__init__()
        if rank is None:
            rank = n_dims
        self.n_dims = n_dims
        self.rank = rank
        self.eps = eps
        self.A = nn.Parameter(torch.randn(n_dims, rank))

    def forward(self, *args):
        return self.A @ self.A.T + torch.eye(self.n_dims) * self.eps

    @property
    def value(self):
        return self()


# TODO: generalize this so that positiveness can arise from other functions
class PositiveDiagonal(nn.Module):
    def __init__(self, n_dims, eps=1e-16):
        super().__init__()
        self.n_dims = n_dims
        self.eps = eps
        self.D = nn.Parameter(torch.randn(n_dims))

    def forward(self, *args):
        return torch.diag(self.D**2 + self.eps)

    @property
    def value(self):
        return self()
