"""Script for large-scale advection analysis
"""
import numpy as np
import xarray as xr

from util import wrap_xarray_calculation


def zadv_numpy(phi, w, dz, axis=-1):
    phi = phi.swapaxes(axis, -1)
    w = w.swapaxes(axis, -1)

    F = np.roll(w, -1, axis=-1)*(np.roll(phi, -1, axis=-1)-phi)/dz
    out =  .5*(np.roll(F, 1, axis=-1) + F)

    return out.swapaxes(axis, -1)


def zadv(phi, w):
    z = phi.z.values
    zc = np.hstack((-z[0], z, 2*z[-1] - z[-2]))
    zw = (zc[1:]  + zc[:-1])/2
    dz = np.diff(zw)

    out = zadv_numpy(phi.data, w.data, dz, axis=phi.get_axis_num('z'))

    return xr.DataArray(out, phi.coords)


def xadv(phi, u, dim='x'):
    # assume constant resolution
    dx = float(phi[dim][1]-phi[dim][0])
    out = zadv_numpy(phi.data, u.data, dx, axis=phi.get_axis_num(dim))
    return xr.DataArray(out, phi.coords)

def advection(phi, u, w):
    return xadv(phi, u) + zadv(phi, w)


def test_advection():
    # setup c-grid
    nx = 200
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    x = xr.DataArray(x, {'x': x})
    xc = float(x[1] - x[0])/2 + x

    # z is at the cell center
    nz = 64
    z = np.linspace(0, np.pi, nz+1)
    zh = (z[1:] + z[:-1])/2
    zw = xr.DataArray(z[:-1], {'z': zh})
    zc = xr.DataArray(zh, {'z': zh})

    u = np.cos(zc) * np.sin(x)
    phi = zc * np.sin(xc)
    w=  np.sin(zw) + 0 * x

    # z advection
    exact = np.sin(zc) * np.sin(xc)
    approx = zadv(phi, w)
    np.testing.assert_allclose(exact.values, approx.values, atol=1e-3)

    # x advection
    exact = np.cos(zc) *np.sin(xc) * zc*np.cos(xc)
    approx = xadv(phi, u)
    np.testing.assert_allclose(exact.values, approx.values, atol=1e-3)


def main_snakemake():
    i = snakemake.input
    out = wrap_xarray_calculation(advection)(i.phi, i.u, i.w)
    out.to_netcdf(snakemake.output[0])

try:
    snakemake
except NameError:
    print("No snakemake object...exiting")
    import sys
    sys.exit(-1)

main_snakemake()
