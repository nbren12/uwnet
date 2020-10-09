import numpy as np
import dask.array as da
import xarray as xr

from .datasets import tiltwave
from .xcalc import centderiv, centspacing, dask_centdiff

PLOT = False


def test_dask_centderiv():
    x, y = np.mgrid[0:2 * np.pi:10j, 0:2 * np.pi:10j]

    z = da.from_array(x, chunks=(10, 10))
    dz = dask_centdiff(z, axis=0, boundary='periodic')

    dx = np.roll(x, -1, 0) - np.roll(x, 1, 0)
    np.testing.assert_allclose(dx, dz.compute())


def test_centdiff():

    A = tiltwave().chunk()

    B = centderiv(A, dim='z')
    B = A.centderiv(dim='z')

    if PLOT:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fig, (a, b) = plt.subplots(1, 2)
        B.plot(ax=a)

        A.plot(ax=b)
        plt.show()


def test_centspacing():
    x = np.arange(5)

    xd = xr.DataArray(x, coords=(('x', x), ))

    dx = centspacing(xd).values

    np.testing.assert_allclose(dx, np.ones_like(dx) * 2)
