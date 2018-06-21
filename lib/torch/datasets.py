"""Torch dataset classes for working with multiscale sam data
"""
import attr
import numpy as np
import torch
from torch.utils.data import Dataset
from toolz import valmap


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class DictDataset(Dataset):
    def __init__(self, mapping):
        self.datasets = mapping

    def __getitem__(self, i):
        return {key: val[i] for key, val in self.datasets.items()}

    def __len__(self):
        return min(len(d) for d in self.datasets.values())


@attr.s
class WindowedData(Dataset):
    """Window data along first dimension

    Examples
    --------
    >>> arr = np.arange(10).reshape((10, 1, 1, 1))
    >>> d = WindowedData(arr)
    >>> d[0:5][:,0,0]
    array([0, 1, 2])
    >>> d[0:5][:,1,0]
    array([1, 2, 3])
    >>> d[0:5][:,2,0]
    array([2, 3, 4])
    >>> d[0:5][:,3,0]
    array([3, 4, 5])
    """
    x = attr.ib()
    chunk_size = attr.ib(default=3)

    @property
    def nwindows(self):
        t  = self.reshaped.shape[0]
        return  t - self.chunk_size + 1

    @property
    def reshaped(self):
        sh  = self.x.shape

        nt = sh[0]
        if self.x.ndim == 4:
            nf = sh[-1]
        elif self.x.ndim == 3:
            nf = 1
        else:
            raise ValueError("data has dimension less the 3. Maybe it is not a time series variable.")
        return self.x.reshape((nt, -1, nf))

    def __len__(self):
        b = self.reshaped.shape[1]
        return self.nwindows * b


    def __getitem__(self, ind):
        """i + j nt  = ind

        """
        i = ind % self.nwindows
        j = (ind-i) // self.nwindows

        return self.reshaped[i:i+self.chunk_size,j,:]

    def __repr__(self):
        return "WindowedData()"


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
