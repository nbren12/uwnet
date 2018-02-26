import xarray as xr
from scipy.interpolate import interp1d


def interp_np(x, pold, pnew, axis=-1):
    return interp1d(pold, x, axis=axis, bounds_error=False)(pnew)


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
