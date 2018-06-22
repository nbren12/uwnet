import click
from toolz import valmap

import torch
import xarray as xr
from torch.utils.data import Dataset
from uwnet import thermo


def _stack_or_rename(x, **kwargs):
    for key, val in kwargs.items():
        if isinstance(val, str):
            x = x.rename({val: key})
        else:
            x = x.stack(**{key: val})
    return x


def _ds_slice_to_numpy_dict(ds):
    dim_order = ['xbatch', 'xtime', 'xfeat']
    out = {}
    for key in ds.data_vars:
        dims = [dim for dim in dim_order
                if dim in ds[key].dims]
        out[key] = ds[key].transpose(*dims).values

    return out


def _ds_slice_to_torch(ds):
    return valmap(lambda x: torch.from_numpy(x).detach(),
                  _ds_slice_to_numpy_dict(ds))


class XRTimeSeries(Dataset):
    """A pytorch Dataset class for time series data in xarray format

    Parameters
    ----------
    data : xr.Dataset
        input data
    dims : seq
        list of dimensions used to reshape the data. Format::

            (time_dims, batch_dims, feature_dims)

    Attributes
    ----------
    std
    mean
    scale

    Examples
    --------
    >>> ds = xr.open_dataset("in.nc")
    >>> XRTimeSeries(ds, [['time'], ['x', 'y'], ['z']])

    """

    def __init__(self, data, dims):
        """Initialize XRTimeSeries.

        """
        self.data = data
        self.dims = dims
        self._ds = _stack_or_rename(self.data, xtime=self.dims[0],
                                    xbatch=self.dims[1],
                                    xfeat=self.dims[2])

    def __len__(self):
        res = 1
        for dim in self.dims[1]:
            res *= len(self.data[dim])
        return res

    def __getitem__(self, i):
        ds = self._ds.isel(xbatch=i)
        return _ds_slice_to_numpy_dict(ds)

    @property
    def mean(self):
        """Mean of the contained variables"""
        ds = self._ds.mean(['xbatch', 'xtime'])
        return _ds_slice_to_torch(ds)

    @property
    def std(self):
        """Standard deviation of the contained variables"""
        ds = self._ds.std(['xbatch', 'xtime'])
        return _ds_slice_to_torch(ds)

    @property
    def scale(self):
        std = self.std
        return valmap(lambda x: x.max(), std)


def load_data(paths):
    data = {}
    for info in paths:
        for field in info['fields']:
            ds = xr.open_dataset(info['path'], chunks={'time': 10})
            data[field] = ds[field]

    # compute layer mass from stat file
    rho = data.pop('RHO')[0]
    rhodz = thermo.layer_mass(rho)

    data['layer_mass'] = rhodz

    TABS = data.pop('TABS')
    QV = data.pop('QV')
    QN = data.pop('QN', 0.0)
    QP = data.pop('QP', 0.0)

    sl = thermo.liquid_water_temperature(TABS, QN, QP)
    qt = QV + QN

    data['sl'] = sl
    data['qt'] = qt

    objects = [
        val.to_dataset(name=key).assign(x=sl.x, y=sl.y)
        for key, val in data.items()
    ]
    return xr.merge(objects, join='inner').sortby('time')


def get_dataset(paths, post=None, **kwargs):
    # paths = yaml.load(open("config.yaml"))['paths']
    ds = load_data(paths)
    if post is not None:
        ds = post(ds)
    ds = ds.load()
    return XRTimeSeries(ds, [['time'], ['x', 'y'], ['z']])


@click.command()
@click.argument("out")
def main(out):
    import yaml
    paths = yaml.load(open("config.yaml"))['paths']
    ds = load_data(paths).load()
    ds.to_netcdf('out.nc')


if __name__ == '__main__':
    main()
