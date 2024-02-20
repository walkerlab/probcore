import torch
from torch import nn


class TransformedParameter(nn.Module):
    """
    A module for applying a transformation function to a torch.nn.Parameter.
    This can be useful for ensuring that a parameter adheres to certain constraints
    (e.g., positivity) after transformation. The forward method applies the transformation
    function to the parameter.

    Attributes:
        parameter (nn.Parameter): The parameter to be transformed.
        transform_fn (Callable): The transformation function to be applied to the parameter.
        value (torch.Tensor): The transformed parameter value.

    Args:
        tensor (torch.Tensor): The initial tensor to be wrapped as an nn.Parameter.
        transform_fn (Callable, optional): The transformation function to be applied.
            If None, the identity function is used, meaning no transformation is applied.

    Returns:
        torch.Tensor: The transformed parameter as a tensor, with shape dependent on the
        transformation function and the initial tensor shape.
    """

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
    """
    A module to represent a covariance matrix as a parameterized entity in a neural network.
    This implementation ensures the covariance matrix is positive semi-definite by constructing
    it as A @ A.T + epsilon * I, where A is a parameter matrix, and epsilon is a small positive
    constant added for numerical stability.

    Attributes:
        A (nn.Parameter): The parameter matrix used to construct the covariance matrix.
        eps (float): A small positive constant added to the diagonal for numerical stability.
        value (torch.Tensor): The covariance matrix.

    Args:
        n_dims (int): The dimensionality of the square covariance matrix.
        rank (int, optional): The rank of the matrix A used in constructing the covariance matrix.
            If None, it defaults to n_dims, resulting in a full-rank covariance matrix.

    Returns:
        torch.Tensor: The covariance matrix, with shape (n_dims, n_dims).
    """

    def __init__(self, n_dims, rank=None):
        super().__init__()
        if rank is None:
            rank = n_dims
        self.n_dims = n_dims
        self.rank = rank
        self.A = nn.Parameter(torch.randn(n_dims, rank))
        self.eps = torch.finfo(self.A.dtype).eps

    def forward(self, *args):
        return self.A @ self.A.T + torch.eye(self.n_dims) * self.eps

    @property
    def value(self):
        return self()


# TODO: generalize this so that positiveness can arise from other functions
class PositiveDiagonal(nn.Module):
    """
    A module for representing a diagonal matrix with positive diagonal elements. This is achieved
    by squaring the elements of a parameter vector D and adding a small positive constant epsilon
    to each squared element for numerical stability.

    Attributes:
        D (nn.Parameter): The parameter vector whose squared elements form the diagonal of the matrix.
        eps (float): A small positive constant added to each element of the squared D for numerical stability.
        value (torch.Tensor): The resulting diagonal matrix with positive diagonal elements.

    Args:
        n_dims (int): The dimensionality of the square diagonal matrix.
        eps (float, optional): A small positive constant added for numerical stability. Defaults to 1e-16.

    Returns:
        torch.Tensor: The diagonal matrix with positive diagonal elements, with shape (n_dims, n_dims).
    """

    def __init__(self, n_dims, eps=1e-16):
        super().__init__()
        self.n_dims = n_dims
        self.D = nn.Parameter(torch.randn(n_dims))
        self.eps = torch.finfo(self.D.dtype).eps

    def forward(self, *args):
        return torch.diag(self.D**2 + self.eps)

    @property
    def value(self):
        return self()
