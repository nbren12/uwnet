from glob import glob

import numpy as np

import xarray as xr
from xnoah import centderiv

from .advection import material_derivative
from .thermo import (d_liquid_water_temperature, layer_mass, lhf_to_tendency,
                     liquid_water_temperature, shf_to_tendency, total_water)
from .util import compute_weighted_scale


def prepare_array(x):
    output_dims = [dim for dim in ['time', 'y', 'x', 'z'] if dim in x.dims]
    return x.transpose(*output_dims).values


def prepare_data(inputs: xr.Dataset, forcings: xr.Dataset):

    w = inputs.w

    fields = ['sl', 'qt']

    weights = {key: w.values for key in fields}

    # compute scales
    sample_dims = set(['x', 'y', 'time']) & set(inputs.dims)
    scales = compute_weighted_scale(
        w, sample_dims=sample_dims, ds=inputs[fields])
    scales = {key: float(scales[key]) for key in fields}

    X = {key: prepare_array(inputs[key]) for key in inputs.data_vars}
    G = {key: prepare_array(forcings[key]) for key in forcings.data_vars}

    # return stacked data

    return {
        'X': X,
        'G': G,
        'scales': scales,
        'w': weights,
        'p': inputs.p.values,
        'z': inputs.z.values
    }


def defaultsel(x):
    return x.isel(y=slice(24, 40))


def inputs_and_forcings(file_3d, file_2d, stat_file, sel=defaultsel):

    data_3d = xr.open_dataset(file_3d)
    data_2d = xr.open_dataset(file_2d)
    stat = xr.open_dataset(stat_file)

    data = xr.merge((data_3d, data_2d), join='inner').sortby('time')

    # limit to tropics
    data = sel(data)

    p = stat.p
    rho = stat.RHO[0].drop('time')
    w = layer_mass(rho)

    # compute the surface flux tendencies

    inputs = xr.Dataset({
        'sl':
        liquid_water_temperature(data.TABS, data.QN, data.QP),
        'qt':
        total_water(data.QV, data.QN)
    })

    forcings = inputs.apply(
        lambda f: -material_derivative(data.U, data.V, data.W, f) * 86400)

    forcings['SHF'] = data.SHF
    forcings['LHF'] = data.LHF
    forcings['QRAD'] = data.QRAD
    forcings['Prec'] = data.Prec
    forcings['W'] = data.W
    forcings['SOLIN'] = data.SOLIN

    inputs['p'] = p
    inputs['w'] = w

    return inputs, forcings
