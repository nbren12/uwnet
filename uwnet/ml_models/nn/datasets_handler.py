from dataclasses import dataclass

import numpy as np
import torch
from toolz import valmap
from torch.utils.data import Dataset
from itertools import product
import xarray as xr

from uwnet.utils import dataset_to_broadcastable_array_dict
from uwnet.tensordict import TensorDict
from src.data import assign_apparent_sources

from uwnet.thermo import sec_in_day

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


def _numpy_to_torch(x):
    y = torch.from_numpy(x).detach().float()
    return y.view(-1, 1, 1)


def _ds_slice_to_torch(ds):
    return valmap(_numpy_to_torch, _ds_slice_to_numpy_dict(ds))


class XRTimeSeries(Dataset):
    """A pytorch Dataset class for time series data in xarray format

    This function assumes the data has dimensions ['time', 'z', 'y', 'x'], and
    that the axes of the data arrays are all stored in that order.

    An individual "sample" is the full time time series from a single
    horizontal location. The time-varying variables in this sample will have
    shape (time, z, 1, 1).

    Examples
    --------
    >>> ds = xr.open_dataset("in.nc")
    >>> dataset = XRTimeSeries(ds)
    >>> dataset[0]

    """
    dims = ['time', 'z', 'x', 'y']

    def __init__(self, data, time_length=None):
        """
        Parameters
        ----------
        data : xr.DataArray
            An input dataset. This dataset must contain at least some variables
            with all of the dimensions ['time' , 'z', 'x', 'y'].
        time_length : int, optional
            The length of the time sequences to use, must evenly divide the
            total number of time points.
        """
        self.time_length = time_length or len(data.time)
        self.data = data
        self.numpy_data = {key: data[key].values for key in data.data_vars}
        self.data_vars = set(data.data_vars) - {'p', 'RHO', 'rho', 'Ps', 'layer_mass'}
        self.dims = {key: data[key].dims for key in data.data_vars}
        self.constants = {
            key
            for key in data.data_vars
            if len({'x', 'y', 'time'} & set(data[key].dims)) == 0
        }
        self.setup_indices()

    def setup_indices(self):
        len_x = len(self.data['x'].values)
        len_y = len(self.data['y'].values)
        len_t = len(self.data['time'].values)

        x_iter = range(0, len_x, 1)
        y_iter = range(0, len_y, 1)
        t_iter = range(0, len_t, self.time_length)
        assert len_t % self.time_length == 0
        self.indices = list(product(t_iter, y_iter, x_iter))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t, y, x = self.indices[i]
        output_tensors = {}
        for key in self.data_vars:
            if key in self.constants:
                continue

            data_array = self.numpy_data[key]
            if 'z' in self.dims[key]:
                this_array_index = (slice(t, t + self.time_length),
                                    slice(None), y, x)
            else:
                this_array_index = (slice(t, t + self.time_length), None, y, x)

            sample = data_array[this_array_index][:, :, np.newaxis, np.newaxis]
            output_tensors[key] = sample.astype(np.float32)
        return output_tensors

    @property
    def time_dim(self):
        return self.dims[0][0]

    def torch_constants(self):
        return {
            key: torch.tensor(self.data[key].values, requires_grad=False)
            .float()
            for key in self.constants
        }

    @property
    def scale(self):
        std = self.std
        return valmap(lambda x: x.max(), std)



@dataclass
class XarrayBatchLoader:
    """Yield batches from an xarray"""
    dataset: xr.Dataset
    batch_size: int
    dims: tuple = ('sample', 'time', 'z', 'x', 'y')
    variables: list = None
    torch: bool = False

    @property
    def batches(self):
        n = self.num_samples
        batch_size = self.batch_size
        indices = list(range(0, n, batch_size)) + [n]
        batches = [slice(i_start, i_end)
                   for i_start, i_end in zip(indices[:-1], indices[1:])]
        return batches


    @property
    def num_samples(self):
        return len(self.dataset.sample)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            if self.variables is None:
                variables = list(self.dataset.data_vars)
            else:
                variables = self.variables
            subset = self.dataset[variables].isel(sample=batch)
            numpy_dict = dataset_to_broadcastable_array_dict(subset, dims=self.dims)
            if self.torch:
                yield TensorDict.from_numpy_dict(numpy_dict)
            else:
                yield numpy_dict

def get_timestep(data):
    time_dim = 'time'
    time = data[time_dim]
    dt = np.diff(time)

    all_equal = dt.std() / dt.mean() < 1e-6
    if not all_equal:
        raise ValueError("Data must be uniformly sampled in time")

    if time.units.startswith('d'):
        return dt[0] * 86400
    elif time.units.startswith('s'):
        return dt[0]
    else:
        raise ValueError(
            f"Units of time are {time.units}, but must be either seconds"
            "or days")


def get_dataset(data, predict_radiation):
    # _log.info("Opening xarray dataset")

    try:
        dataset = xr.open_zarr(data)
    except ValueError:
        dataset = xr.open_dataset(data)

    if not predict_radiation:
        dataset['FSLI'] = dataset['FSLI'] + dataset['QRAD'] / sec_in_day

    try:
        return dataset.isel(step=0).drop('step').drop('p')
    except:
        return dataset


def get_data_loader(
    ds: xr.Dataset,
    prognostics, 
    batch_size
):
    # List needed variables
    variables = prognostics + ['SST', 'SOLIN', 'QRAD']
    for variable in prognostics:
        forcing_key = 'F' + variable
        variables.append(forcing_key)

    train_data = XarrayBatchLoader(ds, batch_size=batch_size, variables=variables, torch=True)
    return train_data