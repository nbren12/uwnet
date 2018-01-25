import numpy as np
import xarray as xr
from lib.advection import advection_forcing, vertical_advection_upwind

plot = True

def assert_broadcast_eq(x, y):
    x, y = xr.broadcast(x, y)
    np.testing.assert_array_almost_equal(x.values, y.values)


def test_advection_surface_forcing():

    dims = ['x', 'y', 'z', 'time']
    coords = {dim: np.arange(10) for dim in dims}
    shape = [coords[dim].shape[0] for dim in coords]

    f = xr.Dataset({'f': (dims, np.ones(shape))}, coords=coords)
    f = f.f

    one = 0 *f +1
    zero = 0*f

    rho = f.z*0 + 1



    md = advection_forcing(zero, one, zero, f.x + 0 * f, rho)
    np.testing.assert_array_almost_equal(md.values, 0)

    md = advection_forcing(one, zero, zero, f.x + 0 * f, rho)
    assert_broadcast_eq(md.isel(x=slice(1,-1)),
                        -one.isel(x=slice(1,-1)))

    md = advection_forcing(zero, one, zero, f.y + 0 * f, rho)
    assert_broadcast_eq(md.isel(y=slice(1,-1)),
                        -one.isel(y=slice(1,-1)))

    md = advection_forcing(zero, zero, one, f.z + 0 * f, rho)
    assert_broadcast_eq(md.isel(z=slice(1,-1)),
                        -one.isel(z=slice(1,-1)))


def test_vertical_advection_upwind():
    from numpy import sin, cos, exp
    n = 30
    dz = np.pi/n
    zc = dz/2 + np.arange(n) * dz
    zw = np.arange(n) * dz

    rhoc = np.exp(-zc)
    rhow = np.exp(-zw)

    w = np.sin(zw)/rhow
    f = zc

    truth = lambda z: (sin(z) + z*cos(z)) / exp(-z)

    out = vertical_advection_upwind(w, f, rhoc, dz, axis=0)


    if plot:
        import matplotlib.pyplot as plt
        plt.plot(out)
        plt.plot(truth(zc), label='centered')
        plt.plot(truth(zw), label='zw')
        plt.legend()
        plt.show()


    np.testing.assert_allclose(out, truth(zc), atol=dz*30)
