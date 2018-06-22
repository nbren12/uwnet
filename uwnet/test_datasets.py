import numpy as np

import xarray as xr
from uwnet.data import XRTimeSeries


def test_XRTimeSeries():
    a = xr.DataArray(np.ones((3, 4, 5, 2)))
    b = xr.DataArray(np.zeros((3, 4, 5, 2)))
    c = xr.DataArray(np.zeros((3, 4, 5)))
    ds = xr.Dataset({'a': a, 'b': b, 'c': c})

    o = XRTimeSeries(ds, [['dim_2'], ['dim_0', 'dim_1'], ['dim_3']])
    assert len(o) == 3 * 4

    assert o[0]['a'].shape == (5, 2)
    assert o[0]['c'].shape == (5, )

    o = XRTimeSeries(ds, [['dim_1'], ['dim_0', 'dim_2'], ['dim_3']])
    assert len(o) == 3 * 5

    assert o[0]['a'].shape == (4, 2)
    assert o[0]['c'].shape == (4, )
