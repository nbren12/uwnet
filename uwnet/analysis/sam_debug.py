"""Routines for opening and analyzing debugging output from SAM/UWNET"""
import glob
import logging
import os
from collections.abc import Mapping

import numpy as np
import pandas as pd

import torch
import xarray as xr


def index_like(x, y):
    if isinstance(x, Mapping):

        keys = set(x) & set(y.data_vars)
        return xr.Dataset({key: index_like(x[key], y[key]) for key in keys})
    else:
        if x.shape[0] == 1:
            x = x[0]
        return xr.DataArray(x, dims=y.dims, coords=y.coords)


def open_debug_state_like_ds(path: str, ds: xr.Dataset) -> xr.Dataset:
    """Open SAM debugging output as xarray object

    Parameters
    ----------
    path
        path to pickle saved by torch
    ds
        dataset to use a template

    Returns
    -------
    state
        dataset with fields from path
    """
    dbg = torch.load(path)
    state = index_like(dbg['args'][0], ds)
    out = index_like(dbg['out'], ds)
    out = out.rename(
        {key: key + 'NN'
         for key in set(state.data_vars) & set(out.data_vars)})
    return xr.merge([state, out])


def open_debug_files_as_numpy(files):
    out = {}

    logger = logging.getLogger(__name__)
    for file in files:
        logger.info(f"Opening {file}")
        args = torch.load(file)['args'][0]
        for key in args:
            out.setdefault(key, []).append(args[key])

    for key in out:
        out[key] = np.stack(out[key])

    return out


def expand_dims(x):
    """Assign values to all data-less dimensions

    Needed by hv.Dataset
    """
    coords = {}
    for k, dim in enumerate(x.dims):
        if dim not in x.coords:
            coords[dim] = np.arange(x.shape[k])
    y = x.assign_coords(**coords)
    y.name = x.name
    return y


def concat_datasets(args, name='mode'):
    """Concatenate datasets with a new named index

    This function is especially useful for comparing two datasets with
    shared variables with holoviews

    Parameters
    ----------
    args : list
        list of (name, dataset) pairs
    name : str
        name of the new dimension

    Returns
    -------
    ds : xr.Dataset
        concatenated data
    """

    names, vals = zip(*args)

    # get list of vars
    vars = set(vals[0].data_vars)
    for val in vals:
        vars = vars & set(val.data_vars)
    vars = list(vars)

    vals = [val[vars] for val in vals]

    return xr.concat(vals, dim=pd.Index(names, name=name))


def open_debug_and_training_data(t, ids, training_data_path):
    """Open an concatenate the debugging and training data"""

    debug_files = {
        tag: glob.glob(os.path.join(path, '*.pkl'))
        for tag, path in ids.items()
    }

    # open training
    training_ds = xr.open_dataset(training_data_path)
    train_ds_init_time = training_ds.isel(time=0)

    args = [('Train', train_ds_init_time)]
    for tag in debug_files:
        dbg = open_debug_state_like_ds(debug_files[tag][t], train_ds_init_time)
        args.append((tag, dbg))
    return concat_datasets(args, name='tag')
