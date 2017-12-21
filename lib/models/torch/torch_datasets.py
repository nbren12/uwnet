"""Torch dataset classes for working with multiscale sam data
"""
import warnings

import attr
import numpy as np
from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def _get_windowed_data(x, chunk_size):
    """Get batched windows of x along temporal dimensions

    Parameters
    ----------
    x : (nt, ny, nx, nfeatures)
        array of four dimensional data
    chunk_size : int
        size of windows along time dimension

    """
    import skimage
    nt, ny, nx, nf = x.shape
    view = x.reshape((nt, ny * nx, nf))
    window_shape = (chunk_size, ny * nx, nf)
    windowed = skimage.util.view_as_windows(view, window_shape)
    # remove singleton dimensions in the window
    windowed = windowed[:, 0, 0, ...]
    nt, nwindow, nspatial, nf = windowed.shape

    windowed = windowed.swapaxes(0, 1)
    # combine the temporal and spatial dimensions
    final_shape = (nwindow, nt * nspatial, nf)

    return windowed.reshape(final_shape)


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

    def __len__(self):
        return self.windowed.shape[1]

    @property
    def windowed(self):
        # need to catch annoying errors in _get_windowed_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _get_windowed_data(self.x, self.chunk_size)

    def __getitem__(self, ind):
        return self.windowed[:, ind, :]
