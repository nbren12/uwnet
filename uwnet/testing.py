import numpy as np
import xarray as xr


def assert_tensors_allclose(*args):
    args = [arg.detach().numpy() for arg in args]
    return np.testing.assert_allclose(*args)


def mock_data(shape=(4,5,6,7), init=np.zeros):
    dims = ['time', 'y', 'x', 'z']
    return xr.DataArray(init(shape), dims=dims)
