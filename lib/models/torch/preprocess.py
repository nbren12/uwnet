"""Data loading and preprocessing routines
"""
import numpy as np
import torch
import xarray as xr
from sklearn.externals import joblib
from torch.autograd import Variable

from lib.util import compute_weighted_scale, scales_to_np, weights_to_np

from toolz import pipe
from toolz.curried import valmap


def stacked_data(X):

    sl = np.asarray(X['sl'])
    qt = np.asarray(X['qt'])

    # do not use the moisture field above 200 hPA
    # this is the top 14 grid points for NGAqua
    # ntop = -14
    # qt = qt[..., :ntop].astype(float)
    # sl = sl[..., :ntop].astype(float)

    return np.concatenate((sl, qt), axis=-1)

    return X


def pad_along_axis(x, pad_width, mode, axis):
    pad_widths = [(0, 0)]*x.ndim
    pad_widths[axis] = pad_width
    return np.pad(x, pad_widths, mode)


def _stacked_to_dict(X):
    """Inverse operation of stacked_data """

    nf = X.shape[-1]
    # nz + nz - 14 = nf
    # nz = (nf+14)//2
    nz = nf//2

    sl = X[...,:nz]
    qt = X[...,nz:]

    # qt = pad_along_axis(qt, (0,14), 'constant', -1)

    return {'sl': sl, 'qt': qt}


def stacked_to_xr(X, **kwargs):
    d = _stacked_to_dict(X)
    data_vars = {key: xr.DataArray(val, **kwargs) for key, val in d.items()}
    return xr.Dataset(data_vars)


def _dataset_to_dict(ds: xr.Dataset):
    return {key: ds[key].values for key in ds.data_vars}


def wrap(torch_model):
    def fun(*args):
        torch_args = [pipe(x, _dataset_to_dict,
                           valmap(torch.FloatTensor),
                           valmap(Variable))
                      for x in args]
        y = torch_model(*torch_args)
        y = valmap(lambda x: x.cpu().data.numpy(), y)

        # get coords from inputs
        x = args[0]

        return xr.Dataset(valmap(
            lambda arr: xr.DataArray(arr, coords=x.coords, dims=x.dims),
            y
        ))

    return fun


def prepare_data(inputs: xr.Dataset, forcings: xr.Dataset):

    w = inputs.w

    fields = ['sl', 'qt']

    weights = {key: w.values for key in fields}

    # compute scales
    sample_dims = set(['x', 'y', 'time']) & set(inputs.dims)
    scales = compute_weighted_scale(w, sample_dims=sample_dims,
                                    ds=inputs[fields])
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
        'w': w,
        'p': inputs.p.values,
    }
