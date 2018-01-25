import numpy as np
import xarray as xr
from lib.flux_advection import vertical_advection_upwind

plot = False

def assert_broadcast_eq(x, y):
    x, y = xr.broadcast(x, y)
    np.testing.assert_array_almost_equal(x.values, y.values)


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
