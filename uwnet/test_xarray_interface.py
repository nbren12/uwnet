from unittest.mock import Mock

import numpy as np
import pytest
import torch

import xarray as xr

from .xarray_interface import (
    _array_to_tensor, call_with_xr, _torch_dict_to_dataset)


def init_dataarray(shape):
    dims = ['time', 'z', 'y', 'x']
    coords = {dim: np.arange(n) for dim, n in zip(dims, shape)}
    return xr.DataArray(np.ones(shape), dims=dims, coords=coords)


def test__array_to_tensor_3d_input():
    t, z, y, x = shape = (3, 10, 2, 2)
    arr = init_dataarray(shape)
    out = _array_to_tensor(arr)
    assert out.shape == shape

    # test for 3D data
    two_d_data = arr.isel(z=0)
    out = _array_to_tensor(two_d_data)
    assert out.shape == (t, 1, y, x)


def test_call_with_xr_empty_data_fails():
    shape = (1, 1, 0, 1)
    name = 'a'

    da = init_dataarray(shape).to_dataset(name=name)
    mock = Mock()
    with pytest.raises(
            ValueError, match="'y' dimension of 'a' has length 0"):
        call_with_xr(mock, da)


def test_call_with_xr_no_error():
    shape = (10, 1, 1, 1)
    name = 'a'


    da = init_dataarray(shape).to_dataset(name=name)

    # setup mock object
    mock = Mock()
    a = torch.tensor(da[name].values)
    mock.return_value = {name: a}

    call_with_xr(mock, da)


def test__torch_dict_to_dataset():
    name = 'a'
    shape = (10, 1, 1, 1)
    da = init_dataarray(shape).to_dataset(name=name)
    torch_dict = {name: torch.tensor(da[name].values)}
    _torch_dict_to_dataset(torch_dict, da.coords)
