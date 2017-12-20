"""Data loading and preprocessing routines
"""
import numpy as np
from sklearn.externals import joblib
from lib.util import compute_weighted_scale, weights_to_np, scales_to_np

import xarray as xr


def stacked_data(X):

    sl = np.asarray(X['sl'])
    qt = np.asarray(X['qt'])

    # do not use the moisture field above 200 hPA
    # this is the top 14 grid points for NGAqua
    ntop = -14
    qt = qt[..., :ntop].astype(float)

    return np.concatenate((sl, qt), axis=-1)

    return X


def pad_along_axis(x, pad_width, mode, axis):
    pad_widths = [(0, 0)]*x.ndim
    pad_widths[axis] = pad_width
    return np.pad(x, pad_widths, mode)


def unstacked_data(X):
    """Inverse operation of stacked_data """

    nf = X.shape[-1]
    # nz + nz - 14 = nf
    nz = (nf+14)//2

    sl = X[...,:nz]
    qt = X[...,nz:]

    qt = pad_along_axis(qt, (0,14), 'constant', -1)

    return {'sl': sl, 'qt': qt}

def prepare_data(inputs, forcings, w,
                 subset_fn=lambda x: x.isel(y=slice(24, 40))):

    fields = ('sl', 'qt')

    # load the data if necesary
    if not isinstance(inputs, xr.Dataset):
        inputs = xr.open_mfdataset(inputs, preprocess=subset_fn)
    if not isinstance(forcings, xr.Dataset):
        forcings = xr.open_mfdataset(forcings, preprocess=subset_fn)
    if not isinstance(w, xr.DataArray):
        w = xr.open_dataarray(w)

    weights = {key: w.values for key in fields}

    # compute scales
    sample_dims = set(['x', 'y', 'time']) & set(inputs.dims)
    scales = compute_weighted_scale(w, sample_dims=sample_dims,
                                    ds=inputs)
    scales = {key: float(scales[key]) for key in fields}
    scales = {key: [scales[key]] * inputs[key].z.shape[0]
              for key in fields}


    output_dims = [dim for dim in ['time', 'y', 'x', 'z']
                   if dim in inputs.dims]
    X = {key: inputs[key].transpose(*output_dims) for key in fields}
    G = {key: forcings[key].transpose(*output_dims) for key in fields}

    # return stacked data
    X = stacked_data(X)
    G = stacked_data(G)
    scales = stacked_data(scales)
    w = stacked_data(weights)

    return {
        'X': X,
        'G': G,
        'scales': scales,
        'w': w
    }
