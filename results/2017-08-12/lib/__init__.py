import numpy as np
from scipy.linalg import pinv
import gnl.spline as gs
from gnl.xarray import xr
from glob import glob


def smooth(A, dim, n, per=True):
    x = np.asarray(A[dim])

    L = x[-1]*2 - x[-2] if per else x[-1]
    knots = np.linspace(x[0], L, n)

    if per:
        Bx = gs.psplines(x, knots)
    else:
        Bx = gs.splines(x, knots)

    new_dim = 'poksd'
    S = pinv(Bx) @ Bx
    S = xr.DataArray(S, dims=[dim, new_dim], coords={dim: A[dim], new_dim: np.asarray(A[dim])})
    return A.dot(S).rename({new_dim:dim})


class SamRun(object):
    def __init__(self, rev, root="/scratch/ndb245/Data/SAM6.10.9/JOBS"):
        g = glob("{}/*-{}*".format(root, rev))

        if len(g) == 0:
            raise ValueError("Specified ID not found")
        elif len(g) > 1:
            raise ValueError("Specified ID is not unique")
        else:
            self.rev = g[0]

    @property
    def data2d(self):
        return self.ncglob("OUT_2D")

    @property
    def data3d(self):
        return self.ncglob("OUT_3D")

    @property
    def moments(self):
        return self.ncglob("OUT_MOMENTS")
    @property
    def stats(self):
        return self.ncglob("OUT_STAT")



    def ncglob(self,  path):
        return sorted(glob("{}/{}/*.nc".format(self.rev, path)))
