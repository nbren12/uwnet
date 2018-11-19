import numpy as np
import pytest
import xarray as xr
from torch.utils.data import DataLoader

from uwnet.datasets import XRTimeSeries


def get_obj():

    dims_3d = ['time', 'z', 'y', 'x']
    dims_2d = ['time', 'y', 'x']

    data_3d = np.ones((4, 4, 5, 2))
    data_2d = np.ones((4, 5, 2))

    return xr.Dataset({
        'a': (dims_3d, data_3d),
        'b': (dims_2d, data_2d)
    }), data_3d.shape


@pytest.mark.parametrize('time_length', [1, 2, 4])
def test_XRTimeSeries(time_length):
    ds, (t, z, y, x) = get_obj()
    o = XRTimeSeries(ds, time_length=time_length)
    assert len(o) == t * y * x // time_length


def test_XRTimeSeries_shape():
    ds, (t, z, y, x) = get_obj()

    time_length = 2
    o = XRTimeSeries(ds, time_length=time_length)
    assert o[0]['a'].shape == (time_length, z)
    assert o[0]['b'].shape == (time_length, )

    # get last time point
    o[-1]


def test_DataLoader():
    ds, (t, z, y, x) = get_obj()
    time_length = 2
    batch_size = 2
    o = XRTimeSeries(ds, time_length=time_length)
    dl = DataLoader(o, batch_size=batch_size)
    batch = next(iter(dl))
    assert batch['a'].shape == (batch_size, time_length, z)


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
