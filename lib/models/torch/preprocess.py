"""Data loading and preprocessing routines
"""
from collections import Mapping

import numpy as np
import torch
from toolz import curry
from toolz.curried import valmap
from torch.autograd import Variable

import xarray as xr
from lib.util import compute_weighted_scale


def pad_along_axis(x, pad_width, mode, axis):
    pad_widths = [(0, 0)]*x.ndim
    pad_widths[axis] = pad_width
    return np.pad(x, pad_widths, mode)


def _dataset_to_dict(ds: xr.Dataset):
    return {key: ds[key].values for key in ds.data_vars}


def _wrap_args(args, cuda=False, to_numpy=False):

    wrap = curry(_wrap_args, cuda=cuda, to_numpy=to_numpy)
    if isinstance(args, tuple):
        return tuple(map(wrap, args))
    elif isinstance(args, Mapping):
        return valmap(wrap, args)
    elif isinstance(args, xr.Dataset):
        return {key: wrap(args[key]) for key in args.data_vars}
    elif isinstance(args, xr.DataArray):
        x = args
        # transpose data into correct order
        x = x.transpose('time', 'batch', 'z')
        # turn it into a pytorch variable
        return Variable(torch.FloatTensor(x.values))
    elif isinstance(args, Variable):
        if to_numpy:
            return args.data.cpu().numpy()
        else:
            return args


def wrap(torch_model):
    def fun(*args):
        torch_args = _wrap_args(args)

        y = torch_model(*torch_args)
        y  = _wrap_args(y, to_numpy=True)

        return y

    return fun


def prepare_array(x):
    output_dims = [dim for dim in ['time', 'y', 'x', 'z']
                   if dim in x.dims]
    return x.transpose(*output_dims).values


def prepare_data(inputs: xr.Dataset, forcings: xr.Dataset):

    w = inputs.w

    fields = ['sl', 'qt']

    weights = {key: w.values for key in fields}

    # compute scales
    sample_dims = set(['x', 'y', 'time']) & set(inputs.dims)
    scales = compute_weighted_scale(w, sample_dims=sample_dims,
                                    ds=inputs[fields])
    scales = {key: float(scales[key]) for key in fields}

    X = {key: prepare_array(inputs[key]) for key in inputs.data_vars}
    G = {key: prepare_array(forcings[key]) for key in forcings.data_vars}

    # return stacked data

    return {
        'X': X,
        'G': G,
        'scales': scales,
        'w': weights,
        'p': inputs.p.values,
        'z': inputs.z.values
    }
