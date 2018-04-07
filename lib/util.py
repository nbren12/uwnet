import numpy as np
import xarray as xr
import pandas as pd
from xnoah.data_matrix import unstack_cat


def wrap_xarray_calculation(f):
    def fun(*args, **kwargs):
        new_args = []
        for a in args:
            if isinstance(a, str):
                new_args.append(xopena(a))
            else:
                new_args.append(a)

        return f(*new_args, **kwargs)

    return fun


def compute_dp(p):
    """Compute layer thickness in pressure coordinates"""
    p_ghosted = np.hstack((2 * p[0] - p[1], p, 0))
    p_interface = (p_ghosted[1:] + p_ghosted[:-1]) / 2
    dp = -np.diff(p_interface)

    return dp


def compute_weighted_scale(weight, sample_dims, ds):
    def f(data):
        sig = data.std(sample_dims)
        if set(weight.dims) <= set(sig.dims):
            sig = (
                sig**2 * weight / weight.sum()).sum(weight.dims).pipe(np.sqrt)
        return sig

    return ds.apply(f)


def mul_if_dims_subset(weight, x):
    """If weight.dims are a subset of x then weight"""
    if set(weight.dims) <= set(x.dims):
        return x * weight
    else:
        return x
