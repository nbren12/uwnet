"""Data loading and preprocessing routines
"""
from collections import Mapping

import torch
from toolz import curry
from toolz.curried import valmap
from torch.autograd import Variable

import xarray as xr


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
