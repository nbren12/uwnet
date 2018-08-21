import numpy as np
import pytest
import torch

import xarray as xr

from .interface import (dataarray_to_numpy, numpy_dict_to_torch_dict,
                        numpy_to_dataarray, step_with_numpy_inputs,
                        step_with_xarray_inputs)


def test_step_with_numpy_inputs():
    def step(x, dt):
        return {'x': x['x'], 'prec': x['x'][..., 0:1]}, None

    nz = 10
    shape_3d = (nz, 3, 4)
    shape_2d = (1, 3, 4)

    dt = 0.0
    kwargs = {'x': np.ones((shape_3d))}

    out = step_with_numpy_inputs(step, kwargs, dt=None)
    out['x'].shape == shape_3d
    out['prec'].shape == shape_2d

    np.testing.assert_array_equal(out['x'], kwargs['x'])


def test_numpy_dict_to_torch_dict():
    nz, ny, nx = (10, 4, 5)
    shape_3d = (nz, ny, nx)
    shape_2d = (1, ny, nx)

    x = {'a': np.ones(shape_3d), 'b': np.ones(shape_2d), 'c': np.ones((4, ))}
    y = numpy_dict_to_torch_dict(x)

    # check sizes
    assert y['a'].size() == (ny * nx, nz)
    assert y['b'].size() == (ny * nx, 1)
    assert y['c'].size() == (4, )

    # check that outputs are torch tensors
    for key, val in y.items():
        if not torch.is_tensor(val):
            raise ValueError(key + " is not a torch tensor")


def test_dataarray_to_numpy():
    nx, ny, nz = (10, 11, 12)
    da = xr.DataArray(np.ones((ny, nx, nz)), dims=['y', 'x', 'z'])
    x = dataarray_to_numpy(da)
    assert x.shape == (nz, ny, nx)

    # one dimensional array
    da = xr.DataArray(np.ones((ny, nx)), dims=['y', 'x'])
    x = dataarray_to_numpy(da)
    assert x.shape == (1, ny, nx)


def test_numpy_to_dataarray():
    y = numpy_to_dataarray(np.ones((1, 10, 11)))
    assert y.shape == (10, 11)
    assert y.dims == ('y', 'x')

    y = numpy_to_dataarray(np.ones((22, 10, 11)))
    assert y.shape == (22, 10, 11)
    assert y.dims == ('z', 'y', 'x')


def test_step_with_xarray_inputs():
    def step(x, t):
        return {'a': x['a'], 'b': x['b'][..., 0:1]}, None

    nx, ny, nz = (10, 11, 12)
    a = xr.DataArray(np.ones((ny, nx, nz)), dims=['y', 'x', 'z'])
    b = xr.DataArray(np.ones((ny, nx)), dims=['y', 'x'])

    coords = dict(y=np.r_[:ny], x=np.r_[:nx], z=np.r_[:nz])
    ds = xr.Dataset({'a': a, 'b': b}, coords=coords)

    out_ds = step_with_xarray_inputs(step, ds, 0.0)

    # check that these variables exist in the output
    # maybe i should add more rigorous checks
    print(out_ds.a)
    print(out_ds.b)
