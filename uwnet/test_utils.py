import numpy as np
import torch
from uwnet.utils import stack_dicts, dataarray_to_broadcastable_array
import xarray as xr


def test_stack_dicts():

    n = 10
    ins = [{'a': torch.ones(2, 1)} for i in range(n)]

    out = stack_dicts(ins)
    assert out['a'].size() == (2, n, 1)


def test_dataarray_to_broadcastable_array():

    a, b = 4, 5
    dims = ['a', 'b']
    arr = xr.DataArray(np.ones((a, b)), dims=dims)
    desired_dims = ['c', 'a', 'b']
    expected = (1, a, b)
    nparr = dataarray_to_broadcastable_array(arr, desired_dims)
    assert nparr.shape == expected


    a, b = 4, 5
    dims = ['b', 'a']
    arr = xr.DataArray(np.ones((b, a)), dims=dims)
    expected = (1, a, b)
    nparr = dataarray_to_broadcastable_array(arr, desired_dims)
    assert nparr.shape == expected
