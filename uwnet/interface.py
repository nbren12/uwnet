import torch
import numpy as np
from toolz import curry, valmap, first
import xarray as xr


def load_model_from_path(path):
    from uwnet.model import MLP
    d = torch.load(path)['dict']
    return MLP.from_dict(d)


def xr_to_step_model_args(ds):
    """convert xarray to dict of arguments needed by step_model"""
    out = {}

    for key in ds.data_vars:
        val = ds[key]
        dims = set(val.dims)
        if dims == {'x', 'y'}:
            val = val.transpose('y', 'x').values[np.newaxis]
        elif dims == {'x', 'y', 'z'}:
            val = val.transpose('z', 'y', 'x').values
        elif key in {'layer_mass'}:
            val = val.values
        elif key in {'dt'}:
            val = float(val)

        out[key] = val

    return out


def _numpy_3d_to_torch_flat(x):
    if x.ndim > 1:
        nz = x.shape[0]
        x = x.reshape((nz, -1)).T

    y = torch.from_numpy(x).float()
    y.requires_grad = False
    return y


@curry
def _torch_flat_to_numpy_3d(x, shape):
    """Convert flat torch array to numpy and reshape

    Parameters
    ---------
    x
        (batch, feat) array
    shape : tuple
        The horizontal dimension sizes. Example: (ny, nx)
    """
    x = x.double().numpy()
    x = x.T
    nz = x.shape[0]
    orig_shape = (nz, ) + tuple(shape)
    return x.reshape(orig_shape).copy()


def numpy_dict_to_torch_dict(x):
    """Convert dict of numpy arrays to dict of torch arrays"""
    return valmap(_numpy_3d_to_torch_flat, x)


def get_xy_shape(x):
    for key in x:
        if x[key].ndim == 3:
            return x[key].shape[-2:]


def torch_dict_to_numpy_dict(x, shape):
    return valmap(_torch_flat_to_numpy_3d(shape=shape), x)


def step_with_numpy_inputs(step, x, dt):
    """Step model with numpy inputs

    This method is useful for interfacing with external models such as SAM.

    Parameters
    ----------
    x : dict of numpy arrays
        These should be the same inputs and have the same units as specified
        in self.inputs. However, these arrays should have size (z, y, x).
    dt : float
        the time step in seconds

    Returns
    -------
    out : dict of numpy arrays
        A dict of the outputs listed in self.outputs.
    """
    with torch.no_grad():
        x_t = numpy_dict_to_torch_dict(x)
        out, _ = step(x_t, dt)

    return torch_dict_to_numpy_dict(out, shape=get_xy_shape(x))


def dataarray_to_numpy(x):
    dim_order = [dim for dim in ['z', 'y', 'x'] if dim in x.dims]
    y = x.transpose(*dim_order).values
    if y.ndim == 2:
        y = y[np.newaxis]
    return y


def dataset_to_numpy_dict(x):
    return {key: dataarray_to_numpy(x[key]) for key in x.data_vars}


def numpy_to_dataarray(x, dims=None):
    if dims is None:
        if x.ndim == 3:
            dims = ['z', 'y', 'x']
        else:
            raise ValueError("Input must be three dimensional")

    # remove singleton z dimension for two-dimensional inputs
    if 'z' == dims[0] and x.shape[0] == 1:
        x = x[0]
        dims = dims[1:]

    return xr.DataArray(x, dims=dims)


def step_with_xarray_inputs(step, x, dt):
    """Step model with xarray inputs

    This is useful for debugging

    Parameters
    ----------
    x : xr.Dataset
        These should be the same inputs and have the same units as specified
        in self.inputs. However, these arrays should have size (z, y, x).
    dt : float
        the time step in seconds

    Returns
    -------
    out : xr.Dataset
    """
    y = dataset_to_numpy_dict(x)
    out = step_with_numpy_inputs(step, y, dt)
    return xr.Dataset(valmap(numpy_to_dataarray, out), coords=x.coords)
