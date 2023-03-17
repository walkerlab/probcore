import torch
from torch import nn

def register_to_module(module, field, value):
    if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter):
        # register as buffer
        module.register_buffer(field, value)
    else:
        setattr(module, field, value)

def turn_to_tuple(x):
    """
    Given a value x, turn into a consistent tuple
    * if x is None, return an empty tumple ()
    * if x is a non-tuple value, return as a single-element tuple (x,)
    * if x is already a tuple
    """
    if x is None:
        return ()
    return x if isinstance(x, tuple) else (x,)


def make_args(x, *args, **kwargs):
    if isinstance(x, dict): # TODO: consider making it a Collection.Mapping
        kwargs.update(x)
    elif isinstance(x, tuple):
        args = x + args
    else:
        args = (x,) + args
        
    return args, kwargs