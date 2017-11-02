import numpy as np
import xarray as xr
import pandas as pd
from xnoah.data_matrix import unstack_cat


def mat_to_xarray(mat, coord_dict):
    """Add coordinate information to a numpy array

    Examples
    --------
    mat_to_xarray(mat, {1: pd.Index...})
    """

    coords = []
    for i in range(mat.ndim):
        if i in coord_dict:
            coords.append(coord_dict[i])
        else:
            coords.append(pd.Index(range(mat.shape[i]), name="dim_%d" % i))

    return xr.DataArray(mat, coords)


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


def xopen(name, nt=20):
    return xr.open_dataset(name, chunks=dict(time=nt))\
             .apply(lambda x: x.squeeze())


def xopena(name, nt=20):
    # find variable which isn't p
    f = xr.open_dataset(name, chunks={'time': nt})
    varname = [v for v in f.data_vars if v != 'p'][0]
    return f[varname]


def weighted_r2_score(y_test, y_pred, weight):
    from sklearn.metrics import r2_score
    return r2_score(
        y_test * np.sqrt(weight),
        y_pred * np.sqrt(weight),
        multioutput="uniform_average")


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
            sig = (sig**2 * weight / weight.sum()).sum(weight.dims).pipe(
                np.sqrt)
        return sig

    return ds.apply(f)


def mul_if_dims_subset(weight, x):
    """If weight.dims are a subset of x then weight"""
    if set(weight.dims) <= set(x.dims):
        return x * weight
    else:
        return x


def weights_to_np(w, idx):
    """
    TODO Replace with pandas merging functionality
    """

    def f(i):
        if i < 0:
            return 1.0
        else:
            return float(w.sel(z=idx.levels[1][i]))

    return np.array([f(i) for i in idx.labels[1]])


def scales_to_np(sig, idx):
    """

    TODO replace with pandas merging functionality
    """

    def f(i):
        return float(sig[idx.levels[0][i]])

    return np.array([f(i) for i in idx.labels[0]])


def output_to_xr(y, coords):
    """Create xarray from output

    Parameters
    ----------
    y : matrix
         predictions in matrix format
    coords:
         coordinates of the stacked array
    """

    res = xr.DataArray(y, coords)
    return unstack_cat(res, "features").unstack("samples")


def dict_to_xr(data, dim_name="variable"):
    """Concatenate a dictionary along a new dimensions name
    """
    return xr.concat(data.values(), dim=pd.Index(data.keys(), name=dim_name))
