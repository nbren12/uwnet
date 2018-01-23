from glob import glob
import xarray as xr
from lib.advection import material_derivative
from lib.thermo import liquid_water_temperature, total_water, layer_mass_from_p


def defaultsel(x):
    return x.isel(y=slice(24, 40))


def inputs_and_forcings(files, sel=defaultsel):
    if isinstance(files, str):
        files = glob(files)

    D = xr.open_mfdataset(
        files, chunks={'time': 100}, preprocess=lambda x: x.drop('p'))

    # limit to tropics
    D = sel(D)

    p = xr.open_mfdataset(files[0], chunks={'time': 1}).p.isel(time=0).drop('time')
    w = layer_mass_from_p(p)

    inputs = xr.Dataset({
        'sl': liquid_water_temperature(D.TABS, D.QN, D.QP),
        'qt': total_water(D.QV, D.QN)
    })

    forcings = inputs.apply(
        lambda f: -material_derivative(D.U, D.V, D.W, f) * 86400)

    inputs['p'] = p
    inputs['w'] = w

    return inputs, forcings
