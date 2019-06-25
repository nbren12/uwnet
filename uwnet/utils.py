import torch
from toolz import merge_with, first, merge
import itertools
import dask.bag as db
import dask.array as da
from dask.delayed import delayed
import xarray as xr


def stack_dicts(delayeds):
    return merge_with(lambda x: torch.stack(x, dim=1), delayeds)


def select_time(batch, i):
    out = {}
    for key, val in batch.items():
        if val.dim() == 1:
            out[key] = val
        else:
            out[key] = val[i, :]
    return out


def get_batch_size(batch):
    return first(batch.values()).size(0)


def concat_dicts(delayeds, dim=1):
    return merge_with(lambda x: torch.cat(x, dim=dim), *delayeds)


def batch_to_model_inputs(batch, aux, prog, diag, forcing, constants):
    """Prepare batch for input into model

    This function reshapes the inputs from (batch, time, feat) to (time, z, x,
    y) and includes the constants

    """

    batch = merge(batch, constants)

    def redim(val, num):
        if num == 1:
            return val.t().unsqueeze(-1)
        else:
            return val.permute(1, 2, 0).unsqueeze(-1)

    # entries in inputs and outputs need to be tuples
    data_fields = set(map(tuple, aux + prog + diag))

    out = {key: redim(batch[key], num) for key, num in data_fields}

    for key, num in prog:
        if key in forcing:
            out['F' + key] = redim(batch['F' + key], num)

    return out


def centered_difference(x, dim=-3):
    """Compute centered difference with first order difference at edges"""
    x = x.transpose(dim, 0)
    dx = torch.cat(
        [
            torch.unsqueeze(x[1] - x[0], 0), x[2:] - x[:-2],
            torch.unsqueeze(x[-1] - x[-2], 0)
        ],
        dim=0)
    return dx.transpose(dim, 0)


def get_other_dims(x, dim):
    return set(range(x.dim())) - {dim}


def mean_over_dims(x, dims):
    """Take a mean over a list of dimensions keeping the as singletons"""
    for dim in dims:
        x = x.mean(dim=dim, keepdim=True)
    return x


def mean_other_dims(x, dim):
    """Take a mean over all dimensions but the one specified"""
    other_dims = get_other_dims(x, dim)
    return mean_over_dims(x, other_dims)


def split_by_chunks(dataset):
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield delayed(lambda selection: dataset[selection].load())(selection)


def dataset_to_bag(dataset):
    return db.from_delayed(
        [delayed(lambda x: [x])(chunk) for chunk in split_by_chunks(dataset)])


def getarr(delayed_dataset, key, meta):
    delayed_dataset = delayed_dataset[0]
    delayed_arr = delayed_dataset[key]
    arr = da.from_delayed(delayed_arr, shape=meta.shape, dtype=meta.dtype)
    return xr.DataArray(arr, dims=meta.dims, coords=meta.coords)


def getds(delayed_dataset, meta):
    ds = xr.Dataset({})
    for key in meta:
        ds[key] = getarr(delayed_dataset, key, meta[key])

    return ds


def bag_to_dataset(bag, meta=None, dim='time'):
    if meta is None:
        meta = get_meta(bag)
    arrays = [getds(it, meta).drop(dim) for it in bag.to_delayed()]
    return xr.concat(arrays, dim)


def get_meta(bag):
    return bag.to_delayed()[0][0].compute()


def map_dataset(dataset, fun, dim):
    """Map a function accross the dimensions of a dataset

    Notes
    -----
    Computing a mean with the resulting dataset fails

    """
    chunked = dataset.chunk({dim: 1})
    # We don't really need to use a bag here. a list of delayed objects will do
    bag = dataset_to_bag(chunked)
    transformed = bag.map(fun)
    return bag_to_dataset(transformed)


def dataarray_to_broadcastable_array(
        array : xr.DataArray, dims):
    """Convert list an DataArray to an numpy array with a specified
    dimenionality

    Parameters
    ----------
    arrays: xr.DataArray
    dims : Sequence[str]
        desired ordering of the dimensions
    """
    # transpose data if necessary
    dims_in_array = [dim for dim in dims
                     if dim in array.dims]

    transposed = array.transpose(*dims_in_array)

    # expand null dims
    index = tuple(slice(None) if dim in array.dims else None
                  for dim in dims)

    # get data
    return transposed.data[index]


def dataset_to_broadcastable_array_dict(dataset, dims):
    return {key: dataarray_to_broadcastable_array(dataset[key], dims)
            for key in dataset.data_vars}
