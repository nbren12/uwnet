"""Sample datasets
"""
import numpy as np
import xarray as xr


def tiltwave():
    """Periodic tilted wave"""

    x, y = np.ogrid[0:2*np.pi:201j, 0:np.pi:30j]
    z = np.sin(y)*np.cos(x) + np.cos(x-.5)*np.sin(2*y) + 2


    return xr.DataArray(z.T, dims=['z', 'x'], coords=dict(x=x.ravel(), z=y.ravel()))


def rand3d():
    """Random uniform 3d data"""
    sh = (100, 200, 300)
    dims = ['x', 'y', 'z']

    coords = {}
    for d, n in zip(dims, sh):
        coords[d] = np.arange(n)

    return xr.DataArray(np.random.rand(*sh), dims=dims, coords=coords)
