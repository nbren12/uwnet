from src.data import open_ngaqua
import xarray as xr
from scipy.signal import welch

from xgcm import Grid

import numpy as np
import matplotlib.pyplot as plt

fft = np.fft.fft


def interpolated_velocities(u, v):
    """Interpolate the staggered horizontal velocities to the cell centers

    Use's Ryan Abernathy's xgcm package.

    Parameters
    ----------
    u, v : xr.DataArray
        staggered velocities.

    Returns
    -------
    uint, vint : xr.DataArray
        interpolated velocites on (xc, yc) grid
    """
    # setup grid
    xl = u.x.values
    xc = xl + (xl[1] - xl[0]) / 2

    yl = u.y.values
    yc = yl + (yl[1] - yl[0]) / 2

    u = u.rename({'x': 'xl', 'y': 'yc'}).assign_coords(xl=xl, yc=yc)
    v = v.rename({'x': 'xc', 'y': 'yl'}).assign_coords(yl=yl, xc=xc)

    # combine the data
    ds = xr.Dataset({'u': u, 'v': v})

    # create grid object
    coords = {
        'x': {
            'center': 'xc',
            'left': 'xl'
        },
        'y': {
            'center': 'yc',
            'left': 'yl'
        }
    }
    grid = Grid(ds, periodic=['x'], coords=coords)

    # use grid to interpolate
    uint = grid.interp(ds.u, axis='x')
    vint = grid.interp(ds.v, axis='y', boundary='extend')

    return uint, vint


def horizontal_power_spectrum(u, dim='xc', avg=('yc', 'time')):
    dim_vals = u[dim].values

    d = dim_vals[1] - dim_vals[0]
    fu = fft(u, axis=u.get_axis_num(dim)).mean(u.get_axis_num(avg))
    pw = np.abs(fu)**2

    k = np.fft.fftfreq(n=len(fu), d=d)
    k, pw = k[k >= 0], pw[k >= 0]

    return xr.DataArray(pw, dims=['freq_x'], coords=[k])


def horizontal_welch(u, dim='xc', avg=('yc', 'time')):
    dim_vals = u[dim].values
    d = dim_vals[1] - dim_vals[0]
    k, pw = welch(u, fs=1 / d, axis=u.get_axis_num(dim), scaling='spectrum')
    pw = pw.mean(u.get_axis_num(avg))
    k, pw = k[k >= 0], pw[k >= 0]
    return xr.DataArray(pw, dims=['freq_x'], coords=[k])


def kinetic_energy_spectrum(uint, vint):
    """Compute Kinetic Energy Spectra"""
    pw_u = horizontal_welch(uint)
    pw_v = horizontal_welch(vint)
    return (pw_u + pw_v) / 2.0


def kinetic_energy_spectrum_staggered(*args):
    vels = interpolated_velocities(*args)
    return kinetic_energy_spectrum(*vels)


def autocorr(x: xr.DataArray,
             dim: str,
             base_state_dims=(),
             avg_dims=(),
             n=None) -> xr.DataArray:
    """Auto-correlation of a time series

    Parameters
    ----------
    x : xr.DataArray
        Array to compute auto-correlation of
    dim : str
        Dimensions to copmute auto-correlation over
    base_state_dims : Sequence[str]
        List of dimensions

    Returns
    -------
    autocorrelation
    """

    base_state_dims = (dim, ) + tuple(base_state_dims)
    avg_dims = base_state_dims + tuple(avg_dims)

    if n is None:
        n = len(x[dim])
    lags = np.arange(1, n // 2)

    time = x[dim]
    # assume constant spacing
    dt = float(time[1] - time[0])

    x = x - x.mean(base_state_dims)

    denom = (x * x).sum(avg_dims)

    corrs = []
    for lag in lags:
        shift = {dim: lag}
        xs = x.shift(**shift)
        corr = (xs * x).dropna(dim).sum(avg_dims) / denom
        corrs.append(corr)

    lags = xr.DataArray(
        lags * dt, name='lag', attrs={'units': 'day'}, dims=['lag'])

    return xr.concat(corrs, dim=lags)
