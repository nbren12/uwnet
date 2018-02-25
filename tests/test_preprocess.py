import numpy as np
import xarray as xr
from lib.data import prepare_data


def test_prepare_data(regtest):
    # mock the xarray inputs
    time, z, y, x = np.mgrid[:3, :3, :3, :3]

    dims = ['time', 'z', 'y', 'x']
    coords = {key: np.arange(3) for key in dims}


    inputs = xr.Dataset(
        {
            'qt': (dims, x + y),
            'sl': (dims, x + z)
        }, coords=coords)

    forcings = xr.Dataset(
        {
            'qt': (dims, x * y),
            'sl': (dims, x * z)
        }, coords=coords)

    w = - inputs.z

    inputs = inputs.assign(w=w, p=w)

    data = prepare_data(inputs, forcings)

    print(data, file=regtest)
