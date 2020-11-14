"""Calculus for xarray data
"""
import numpy as np
import xarray as xr

import dask.array as da


def dask_centdiff(x, axis=-1, boundary='reflect'):
    """Compute centered difference using dask
    """

    def fun(x):
        return np.roll(x, -1, axis) - np.roll(x, 1, axis)

    return da.overlap.map_overlap(x, fun, {axis: 1},
                                boundary={axis: boundary},
                                dtype=x.dtype)


def centdiff(A, dim='x', boundary='periodic'):
    if A.chunks is None:
        A = A.chunk()

    return xr.apply_ufunc(
        dask_centdiff,
        A,
        input_core_dims=[['x']],
        output_core_dims=[['x']],
        dask='allowed',
        kwargs={"boundary": boundary}
    )


def centspacing(x):
    """Return spacing of a given xarray coord object
    """

    dx = x.copy()

    x = x.values

    xpad = np.r_[2*x[0]-x[1], x, 2* x[-1] - x[-2]]

    dx = xr.DataArray(xpad[2:] - xpad[:-2], dims=dx.dims)

    return dx


def centderiv(A, dim='x', boundary='periodic'):
    """Compute centered derivative of a data array

    Parameters
    ----------
    A: DataArray
        input dataset
    dim: str
        dimension to take derivative along
    boundary: str
        boundary conditions along dimension. One of 'periodic', 'reflect',
        'nearest', 'none', or an value.

    See Also
    --------
    dask.array.ghost.ghost

    """
    return centdiff(A, dim=dim, boundary=boundary)/centspacing(A[dim])


# monkey patch DataArray class
xr.DataArray.centderiv = centderiv
