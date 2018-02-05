import numpy as np
import xarray as xr
from lib.advection import material_derivative


def assert_xarray_almost_equal(x, y):
    x, y = xr.broadcast(x, y)
    np.testing.assert_array_almost_equal(x.values, y.values)


def test_material_derivative():

    dims = ['x', 'y', 'z', 'time']
    coords = {dim: np.arange(10) for dim in dims}
    shape = [coords[dim].shape[0] for dim in coords]

    f = xr.Dataset({'f': (dims, np.ones(shape))}, coords=coords)
    f = f.f

    one = 0 * f + 1
    zero = 0 * f

    md = material_derivative(zero, one, zero, f.x + 0 * f)
    assert_xarray_almost_equal(md, zero)

    md = material_derivative(one, zero, zero, f.x + 0 * f)
    assert_xarray_almost_equal(
        md.isel(x=slice(1, -1)), one.isel(x=slice(1, -1)))

    md = material_derivative(zero, one, zero, f.y + 0 * f)
    assert_xarray_almost_equal(
        md.isel(y=slice(1, -1)), one.isel(y=slice(1, -1)))

    md = material_derivative(zero, zero, one, f.z + 0 * f)
    assert_xarray_almost_equal(
        md.isel(z=slice(1, -1)), one.isel(z=slice(1, -1)))
