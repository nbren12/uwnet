"""Torch dataset classes for working with multiscale sam data
"""
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
        nf = sh[-1]

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
