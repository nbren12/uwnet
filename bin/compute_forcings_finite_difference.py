#!/usr/bin/env python
import numpy as np

import uwnet.thermo
import xarray as xr
from gnl.calculus import c_grid_advective_tendency


def validate_inputs(ds):
    # vertical velocity should vanish
    mean = ds.W[0].mean(['x', 'y'])
    sig = ds.W[0].std(['x', 'y'])

    M = ds.layer_mass.sum()

    mean_ss = np.sqrt(float((ds.layer_mass * (mean**2)).sum()/M))
    sig_ss = np.sqrt(float((ds.layer_mass * (sig**2)).sum()/M))

    if (mean_ss / sig_ss) > 1e-4:
        raise ValueError(
            "The domain mean of W is larger than acceptable. The typical "
            f"magnitude is {mean_ss} while the standard deviation is {sig_ss}."
        )


def compute_advection_tedency_xarray(g, key):

    # handle grid information
    dz = (uwnet.thermo.get_dz(g.z)).values
    rho = (g.layer_mass / dz).values
    dz, rho = [np.reshape(x, (-1, 1, 1)) for x in [dz, rho]]
    dx = float(g.x[1] - g.x[0])

    # handle 4D inputs
    dim_order = ['time', 'z', 'y', 'x']
    u, v, w, f = [
        g[key].transpose(*dim_order).values for key in ['U', 'V', 'W', key]
    ]

    out = c_grid_advective_tendency(u, v, w, f, dx, dz, rho) * 86400

    out = xr.DataArray(out, dims=g.U.dims, coords=g.U.coords)
    out.attrs['units'] = g[key].attrs.get('units', '1') + '/d'
    return out


if __name__ == '__main__':
    import sys

    data, out = sys.argv[1:]

    g = xr.open_zarr(data)
    validate_inputs(g)
    forcings = xr.Dataset(
        dict(
            FQT=compute_advection_tedency_xarray(g, 'qt'),
            FSL=compute_advection_tedency_xarray(g, 'sl')))

    forcings.to_netcdf(out)
