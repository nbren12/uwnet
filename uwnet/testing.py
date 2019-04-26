import numpy as np
import xarray as xr


def assert_tensors_allclose(*args, atol=1e-7, **kwargs):
    args = [arg.detach().numpy() for arg in args]
    return np.testing.assert_allclose(*args, atol=atol, **kwargs)


def mock_data(shape=(4, 5, 6, 7), init=np.zeros):
    dims = ['time', 'y', 'x', 'z']
    return xr.DataArray(init(shape), dims=dims)
