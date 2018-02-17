import numpy as np
from glob import glob
import xarray as xr
from lib.advection import material_derivative
from xnoah import centderiv
from lib.thermo import (liquid_water_temperature, total_water, layer_mass,
                        shf_to_tendency, lhf_to_tendency,
                        d_liquid_water_temperature)


def defaultsel(x):
    return x.isel(y=slice(24, 40))


def inputs_and_forcings(file_3d, file_2d, stat_file, sel=defaultsel):

    data_3d = xr.open_dataset(file_3d)
    data_2d = xr.open_dataset(file_2d)
    data_2d = data_2d.isel(time=np.argsort(data_2d.time.values))
    stat = xr.open_dataset(stat_file)

    data = xr.merge((data_3d, data_2d), join='inner')

    # limit to tropics
    data = sel(data)

    p = stat.p
    rho = stat.RHO[0].drop('time')
    w = layer_mass(rho)

    # compute the surface flux tendencies

    inputs = xr.Dataset({
        'sl': liquid_water_temperature(data.TABS, data.QN, data.QP),
        'qt': total_water(data.QV, data.QN)
    })



    forcings = xr.Dataset({
        'sl': d_liquid_water_temperature(data.divTABS, data.divQN, data.divQP)*86400,
        'qt': total_water(data.divQV, data.divQN) * 86400
    })

    forcings['SHF'] = data.SHF
    forcings['LHF'] = data.LHF
    forcings['QRAD'] = data.QRAD
    forcings['Prec'] = data.Prec
    forcings['W'] = data.W
    forcings['SOLIN'] = data.SOLIN

    inputs['p'] = p
    inputs['w'] = w

    return inputs, forcings
