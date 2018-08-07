import click
import numpy as np
import torch
from dask.diagnostics import ProgressBar
from toolz import valmap
from torch.utils.data import Dataset

import xarray as xr
from uwnet import thermo


def _stack_or_rename(x, **kwargs):
    for key, val in kwargs.items():
        if isinstance(val, str):
            x = x.rename({val: key})
        else:
            x = x.stack(**{key: val})
    return x


def _ds_slice_to_numpy_dict(ds):
    out = {}
    for key in ds.data_vars:
        out[key] = _to_numpy(ds[key])
    return out


def _to_numpy(x: xr.DataArray):
    dim_order = ['xbatch', 'xtime', 'xfeat']
    dims = [dim for dim in dim_order if dim in x.dims]
    return x.transpose(*dims).values


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
        self._ds = _stack_or_rename(
            self.data,
            xtime=self.dims[0],
            xbatch=self.dims[1],
            xfeat=self.dims[2])

    def __len__(self):
        res = 1
        for dim in self.dims[1]:
            res *= len(self.data[dim])
        return res

    def __getitem__(self, i):

        # convert i to an array
        # this code should handle i = slice, list, etc
        i = np.arange(len(self))[i]
        scalar_idx = i.ndim == 0
        if scalar_idx:
            i = [i]

        # get coordinates using np.unravel_index
        # this code should probably be refactored
        batch_dims = self.dims[1]
        batch_shape = [len(self.data[dim]) for dim in batch_dims]

        idxs = np.unravel_index(i, batch_shape)
        coords = {}
        for key, idx in zip(batch_dims, idxs):
            coords[key] = xr.DataArray(idx, dims='xbatch')

        # select, load, and stack the batch
        batch_ds = self.data.isel(**coords).load()
        ds_r = _stack_or_rename(
            batch_ds, xtime=self.dims[0], xfeat=self.dims[2])

        # prepare for output
        out = {}
        for key in ds_r.data_vars:
            if key in self.constants():
                continue

            arr = _to_numpy(ds_r[key])
            arr = torch.from_numpy(arr)
            arr.requires_grad = False
            if scalar_idx:
                arr = arr[0]

            out[key] = arr.float()

        return out

    @property
    def time_dim(self):
        return self.dims[0][0]

    def constants(self):
        for key in self.data:
            if self.time_dim not in self.data[key].dims:
                yield key

    def torch_constants(self):
        return {
            key: torch.tensor(self.data[key].values, requires_grad=False)
            .float()
            for key in self.constants()
        }

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

    def timestep(self):
        time_dim = self.dims[0][0]
        time = self.data[time_dim]
        dt = np.diff(time)

        all_equal = dt.std() / dt.mean() < 1e-6
        if not all_equal:
            raise ValueError("Data must be uniformly sampled in time")

        return dt[0]


def compute_radiation(data):
    lw = data.pop('LWNS')
    sw = data.pop('SWNS')
    data['RADSFC'] = lw - sw

    lw = data.pop('LWNT')
    sw = data.pop('SWNT')
    data['RADTOA'] = lw - sw

    return data


def compute_forcings(data):
    pass


def load_data(paths):

    data = {}
    for info in paths:
        for field in info['fields']:
            try:
                ds = xr.open_dataset(info['path'], chunks={'time': 40})
            except ValueError:
                ds = xr.open_dataset(info['path'])

            data[field] = ds[field]

    # compute layer mass from stat file
    # remove presssur
    rho = data.pop('RHO')[0]
    rhodz = thermo.layer_mass(rho)

    data['layer_mass'] = rhodz

    # compute thermodynamic variables
    TABS = data.pop('TABS')
    QV = data.pop('QV')
    QN = data.pop('QN', 0.0)
    QP = data.pop('QP', 0.0)

    sl = thermo.liquid_water_temperature(TABS, QN, QP)
    qt = QV + QN

    data['sl'] = sl.assign_attrs(units='K')
    data['qt'] = qt.assign_attrs(units='g/kg')

    data = compute_radiation(data)
    compute_forcings(data)

    # rename staggered dimensions
    data['U'] = (data['U']
                 .rename({'xs': 'x', 'yc': 'y'}))

    data['V'] = (data['V']
                 .rename({'xc': 'x', 'ys': 'y'}))

    objects = [
        val.to_dataset(name=key).assign(x=sl.x, y=sl.y)
        for key, val in data.items()
    ]

    ds = xr.merge(objects, join='inner').sortby('time')
    return ds




def get_dataset(paths, post=None, **kwargs):
    # paths = yaml.load(open("config.yaml"))['paths']
    ds = load_data(paths)
    if post is not None:
        ds = post(ds)
    return XRTimeSeries(ds, [['time'], ['x', 'y'], ['z']])


@click.command()
@click.argument("config")
@click.argument("out")
def main(config, out):
    import yaml
    paths = yaml.load(open(config))['paths']
    with ProgressBar():
        load_data(paths)\
            .load()\
            .to_zarr(out)


if __name__ == '__main__':
    main()
