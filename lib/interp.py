import xarray as xr
import metpy.calc as mc
from scipy.interpolate import interp1d


def interp_np(x, pold, pnew, axis=-1):
    return interp1d(pold, x, axis=axis, bounds_error=False)(pnew)


def interp(z, z3, x, old_dim='p', new_dim=None,
           log=False):
    if not new_dim:
        new_dim = old_dim

    z3, x = xr.broadcast_arrays(z3, x)

    if log:
        _interp = mc.log_interp
    else:
        _interp = mc.interp

    val = _interp(z.values, z3.values, x.values, axis=x.get_axis_num(old_dim))
    coords = {}
    coords.update(x.coords)
    coords.pop(old_dim)
    coords[new_dim] = z.values

    dims = [{old_dim: new_dim}.get(dim, dim) for dim in x.dims]
    return xr.DataArray(val, coords=coords, dims=dims)


def pressure_interp(x, pnew, pressure_name='p'):
    val = interp_np(x.values, x[pressure_name], pnew,
                    axis=x.get_axis_num(pressure_name))
    coords = {}
    coords.update(x.coords)
    coords[pressure_name] = pnew
    dims = x.dims
    return xr.DataArray(val, coords=coords, dims=dims)


def pressure_interp_ds(ds, pnew, pressure_name='p'):
    def f(x):
        if pressure_name in x.dims:
            return pressure_interp(x, pnew,
                                   pressure_name=pressure_name)
        else:
            return x

    return ds.apply(f)
