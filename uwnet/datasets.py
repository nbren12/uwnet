import numpy as np
import torch
from toolz import valmap
from torch.utils.data import Dataset

import xarray as xr


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
