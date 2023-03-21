import torch
from torch import nn


# a helper function to visit the target field
# and invoke the field with cond if it is a nn.Module.
# Otherwise, simply return the field content
def invoke_with_cond(attr, cond=None):
    if isinstance(attr, nn.Module):
        attr = attr(*cond)
    return attr


def register_to_module(module, field, value):
    if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter):
        # register as buffer
        module.register_buffer(field, value)
    else:
        setattr(module, field, value)


def turn_to_tuple(x):
    """
    Given an argument x, construct a tuple such that:
    * if x is None, returns an empty tuple ()
    * if x is a non-tuple value, return a single element tuple (x,)
    * if x is already a tuple, simply return x

    Args:
        x (any): An argument to be transformed into a tuple as necessary.

    Returns:
        tuple: A tuple of arguments. Refer to the summary above for how the argument is handled.
    """

    if x is None:
        return ()
    return x if isinstance(x, tuple) else (x,)


def make_args(x, *args, **kwargs):
    if isinstance(x, dict):  # TODO: consider making it a Collection.Mapping
        kwargs.update(x)
    elif isinstance(x, tuple):
        args = x + args
    else:
        args = (x,) + args

    return args, kwargs
