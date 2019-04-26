from functools import partial

import torch
import xarray as xr
from toolz import pipe
from .tensordict import TensorDict


def _is_at_least_2d(da):
    return set(da.dims) >= {'x', 'y'}


def _array_to_tensor(da):
    if _is_at_least_2d(da):
        if 'z' not in da.dims:
            da = da.expand_dims('z', -3)
    return torch.from_numpy(da.values)\
                .detach()\
                .float()


def dataset_to_torch_dict(ds):
    return TensorDict({
        key: _array_to_tensor(ds[key])
        for key in ds.data_vars
    })


def _torch_dict_to_dataset(output, coords):
    # parse output
    data_vars = {}

    dims_2d = [dim for dim in ['time', 'y', 'x'] if dim in coords.dims]
    dims_3d = [dim for dim in ['time', 'z', 'y', 'x'] if dim in coords.dims]

    time_dim_present = 'time' in dims_2d

    for key, val in output.items():
        z_dim = 1 if time_dim_present else 0
        nz = val.size(z_dim)

        if nz == 1:
            data_vars[key] = (dims_2d, output[key].numpy().squeeze(z_dim))
        else:
            data_vars[key] = (dims_3d, output[key].numpy())

    # prepare coordinates
    coords = dict(coords.items())
    return xr.Dataset(data_vars, coords=coords)


def _assert_no_null_dimensions(ds):
    for key in ds.data_vars:
        for k, n in enumerate(ds[key].shape):
            if ds[key].shape[k] == 0:
                dim = ds[key].dims[k]
                raise ValueError(
                    f"'{dim}' dimension of '{key}' has length 0"
                )


def call_with_xr(self, ds, **kwargs):
    """Call the neural network with xarray inputs"""
    _assert_no_null_dimensions(ds)
    tensordict = dataset_to_torch_dict(ds)
    with torch.no_grad():
        output = self(tensordict, **kwargs)
    return _torch_dict_to_dataset(output, ds.coords)


class XRCallMixin(object):
    """PyTorch module for predicting Q1, Q2 and maybe Q3"""
    call_with_xr = call_with_xr
    predict = call_with_xr


class XarrayWrapper(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return call_with_xr(self.model, *args, **kwargs)
