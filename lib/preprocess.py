"""Data loading and preprocessing routines
"""
import numpy as np
from sklearn.externals import joblib
from xnoah.data_matrix import stack_cat
from lib.util import compute_weighted_scale, weights_to_np, scales_to_np

import xarray as xr


def _stack_dict_arrs(d, keys, axis=-1):
    return np.concatenate([d[key] for key in keys], axis=-1)


def _prepvar(X, feature_dims=['z'], sample_dims=['time', 'x', 'y']):
    # select only the tropics
    return stack_cat(X, "features", ['z'])


def stacked_data(X, fields):

    X = _stack_dict_arrs(X, fields)

    # do not use the moisture field above 200 hPA
    # this is the top 14 grid points for NGAqua
    ntop = -14
    X = X[..., :ntop].astype(float)
    return X


def prepare_data(input_files, forcing_files, weight_file,
                 subset_fn=lambda x: x.isel(y=slice(24, 40))):

    fields = ('sl', 'qt')

    # load the data
    inputs = xr.open_mfdataset(input_files)
    forcings = xr.open_mfdataset(forcing_files)

    if subset_fn is not None:
        inputs = inputs.pipe(subset_fn)
        forcings = forcings.pipe(subset_fn)

    # get weights
    w = xr.open_dataarray(weight_file)
    weights = {key: w.values for key in fields}

    # compute scales
    sample_dims = ['x', 'y', 'time']
    scales = compute_weighted_scale(w, sample_dims=sample_dims,
                                    ds=inputs)
    scales = {key: float(scales[key]) for key in scales}
    scales = {key: [scales[key]] * inputs[key].z.shape[0]
              for key in scales}


    output_dims = ['time', 'y', 'x', 'z']
    X = {key: inputs[key].transpose(*output_dims).values for key in fields}
    G = {key: forcings[key].transpose(*output_dims).values for key in fields}

    # return stacked data
    X = stacked_data(X, fields)
    G = stacked_data(G, fields)
    scales = stacked_data(scales, fields)
    w = stacked_data(weights, fields)

    return {
        'X': X,
        'G': G,
        'scales': scales,
        'w': w
    }
