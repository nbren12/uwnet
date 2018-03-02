"""Data loading and preprocessing routines
"""
from collections import Mapping

from toolz import curry, dissoc
from toolz.curried import valmap

import torch
import xarray as xr
from torch.autograd import Variable


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
        y = _wrap_args(y, to_numpy=True)

        return y

    return fun


def _dataset_to_dict(dataset):
    out = {}
    for key in dataset.data_vars:
        x = dataset[key]
        if 'z' not in x.dims:
            x = x.expand_dims('z')
        if 'batch' not in x.dims:
            x = x.expand_dims('batch')

        transpose_dims = [
            dim for dim in ['time', 'batch', 'z'] if dim in x.dims
        ]
        x = x.transpose(*transpose_dims)
        x = Variable(torch.FloatTensor(x.values))
        out[key] = x
    return out


def column_run(model, prognostic, forcing,
               batch_dims=('x', 'y')):

    batch_dims = [dim for dim in batch_dims if dim in prognostic.dims]
    if batch_dims:
        prognostic = prognostic.stack(batch=batch_dims)
        forcing = forcing.stack(batch=batch_dims)

    prog = _dataset_to_dict(prognostic)
    forcing = _dataset_to_dict(forcing)

    prog.pop('p')
    w = prog.pop('w')

    z = Variable(torch.FloatTensor(prognostic.z.values))

    input_data = {
        'prognostic': prog,
        'forcing': forcing,
        'constant': {
            'w': w,
            'z': z
        }
    }

    model.eval()
    y = model(input_data)

    coords = {'z': prognostic['z'], 'time': prognostic['time']}
    if 'batch' in prognostic.dims:
        coords['batch'] = prognostic.batch

    dims = ['time', 'batch', 'z']

    progs = {
        key: xr.DataArray(
            y['prognostic'][key].data.numpy(), coords=coords, dims=dims).unstack('batch')
        for key in y['prognostic']
    }

    coords['time'] = coords['time'][1:]
    prec = xr.DataArray(
        y['diagnostic']['Prec'].data.numpy()[..., 0],
        coords=dissoc(coords, 'z'),
        dims=['time', 'batch']).unstack('batch')

    return xr.Dataset(progs), prec


def rhs(model, prognostic, forcing,
        batch_dims=('time', 'x', 'y')):
    """Xarray wrapper for computing the source terms"""

    batch_dims = [dim for dim in batch_dims if dim in prognostic.dims]
    prognostic = prognostic.stack(batch=batch_dims)
    forcing = forcing.stack(batch=batch_dims)

    prog = _dataset_to_dict(prognostic)
    forcing = _dataset_to_dict(forcing)

    prog.pop('p')
    w = prog.pop('w')

    model.eval()
    y, prec = model.rhs(prog, forcing, w)

    coords = {'z': prognostic['z'], 'batch': prognostic['batch']}
    dims = ['batch', 'z']
    y = {
        key: xr.DataArray(
            val.data.numpy(), coords=coords, dims=dims)
        for key, val in y.items()
    }

    return xr.Dataset(y).unstack('batch')


