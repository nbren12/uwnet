import torch
from torch.autograd import Variable
from torch import nn


def apply_linear_constraint(lin, a, x, *args, inequality=False, v=None,
                            **kwargs):
    """Apply a linear constraint

    Parameters
    ----------
    lin : Callable
        the linear constraint of the form lin(x, *args, **kwargs) = a that is linear in x
    inequality : bool
        assume that lin(x, *args, **kwargs) >= a

    Returns
    -------
    y : Tensor
      x transformed to satisfy the constraint

    """

    val_x = lin(x, *args, **kwargs)


    # use a constant adjustment
    # x - alpha * v
    if v is None:
        v = Variable(torch.ones(x.size(-1)))
    val_v = lin(v, *args, **kwargs)
    alpha = (val_x - a)

    if inequality:
        alpha = alpha.clamp(max=0)

    return x - v * (alpha/val_v)
