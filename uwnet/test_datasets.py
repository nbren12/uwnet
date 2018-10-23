import numpy as np
import pytest
import xarray as xr

from uwnet.datasets import XRTimeSeries


def get_obj():

    dims_3d = ['time', 'z', 'y', 'x']
    dims_2d = ['time', 'y', 'x']

    data_3d = np.ones((3, 4, 5, 2))
    data_2d = np.ones((3, 5, 2))

    return xr.Dataset({
        'a': (dims_3d, data_3d),
        'b': (dims_2d, data_2d)
    }), data_3d.shape


def test_XRTimeSeries():
    ds, (t, z, y, x) = get_obj()

    o = XRTimeSeries(ds)
    assert o[0]['a'].shape == (t, z)
    assert o[0]['b'].shape == (t, )


@pytest.mark.parametrize('dt,units,dt_seconds', [
    (.125, 'd', 10800.0),
    (10, 's', 10.0),
])
def test_XRTimeSeries_timestep(dt, units, dt_seconds):
    ds, _ = get_obj()
    ds = ds.assign_coords(time=np.arange(len(ds['time'])) * dt)
    ds['time'].attrs['units'] = units

    # dim_1 is the time dimension here
    dataset = XRTimeSeries(ds)
    assert dataset.timestep() == dt_seconds

    # non-uniformly sampled data
    time = np.arange(len(ds['time']))**2
    ds = ds.assign_coords(time=time)
    dataset = XRTimeSeries(ds)

    with pytest.raises(ValueError):
        dataset.timestep()


@pytest.mark.xfail()
def test_XRTimeSeries_torch_constants():
    ds = get_obj()

    # dim_1 is the time dimension here
    dataset = XRTimeSeries(ds, [['dim_1'], ['dim_0', 'dim_2'], ['dim_3']])
    assert 'layer_mass' in dataset.torch_constants()

    # constants should not be in batch
    assert 'layer_mass' not in dataset[0]
