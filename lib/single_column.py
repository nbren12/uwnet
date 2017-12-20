import numpy as np
import torch
from toolz import curry
from .models.torch_models import predict

def runsteps(step, x, n):
    """Perform n steps using"""
    out = np.zeros((n, x.shape[0]), dtype=x.dtype)
    out[0] = x

    for i in range(n):
        x = step(x, i)
        out[i] = x

    return out


@curry
def step(net, x, n):
    return predict(net, x[None,:])[0,:]


@curry
def forced_step(net, g, x, n):

    if n < g.shape[0]-1:
        f = (g[n] + g[n+1])/2
    else:
        f = g[n]

    return predict(net, x[None,:])[0,:] + f * 3/24
