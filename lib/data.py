import numpy as np
from glob import glob
import xarray as xr
from lib.advection import material_derivative
from lib.thermo import liquid_water_temperature, total_water, layer_mass


def defaultsel(x):
    return x.isel(y=slice(24, 40))


def inputs_and_forcings(files_3d, file_2d, stat_file, sel=defaultsel):
    if isinstance(files_3d, str):
        files_3d = glob(files_3d)

    data_3d = xr.open_mfdataset(files_3d, preprocess=lambda x: x.drop('p'))
    data_2d = xr.open_dataset(file_2d)
    data_2d = data_2d.isel(time=np.argsort(data_2d.time.values))
    stat = xr.open_dataset(stat_file)

    D = xr.merge((data_3d, data_2d), join='inner')

    # limit to tropics
    D = sel(D)

    p = stat.p
    rho = stat.RHO[0].drop('time')
    w = layer_mass(rho)

    inputs = xr.Dataset({
        'sl': liquid_water_temperature(D.TABS, D.QN, D.QP),
        'qt': total_water(D.QV, D.QN)
    })

    forcings = inputs.apply(
        lambda f: -material_derivative(D.U, D.V, D.W, f) * 86400)

    inputs['p'] = p
    inputs['w'] = w

    return inputs, forcings
