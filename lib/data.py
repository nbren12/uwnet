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


def inputs_and_forcings(file_3d, file_2d, stat_file):

    data_3d = xr.open_dataset(file_3d)
    data_2d = xr.open_dataset(file_2d)
    stat = xr.open_dataset(stat_file)

    data = xr.merge((data_3d, data_2d), join='inner').sortby('time')
    p = stat.p
    rho = stat.RHO[0].drop('time')
    w = layer_mass(rho)

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


def inputs_and_forcings_sam(file_3d, file_2d, stat_file):

    data_3d = xr.open_dataset(file_3d, chunks={'time': 10})
    # patch x coordinate from data_3d onto data_2d
    data_2d = xr.open_dataset(file_2d)\
                .sel(time=data_3d.time)\
                .assign_coords(x=data_3d.x, y=data_3d.y)

    stat = xr.open_dataset(stat_file)

    p = stat.p
    rho = stat.RHO[0].drop('time')
    w = layer_mass(rho)

    sl = liquid_water_temperature(data_3d.TABS, data_3d.QN, data_3d.QP)
    qt = total_water(data_3d.QV, data_3d.QN)
    sl.persist()
    qt.persist()

    forcings = {}
    forcings['qt'] = (qt.sel(step=1) - qt.sel(step=0))/data_3d.dt
    forcings['sl'] = (sl.sel(step=1) - sl.sel(step=0))/data_3d.dt
    forcings['SHF'] = data_2d.SHF
    forcings['LHF'] = data_2d.LHF
    forcings['QRAD'] = data_3d.QRAD.sel(step=0)
    forcings['Prec'] = data_2d.Prec
    forcings['SOLIN'] = data_2d.SOLIN
    forcings['W'] = data_3d.W.sel(step=0)

    inputs = {}
    inputs['p'] = p
    inputs['w'] = w
    inputs['qt'] = qt.sel(step=0)
    inputs['sl'] = sl.sel(step=0)

    inputs = xr.Dataset(inputs)
    forcings = xr.Dataset(forcings)

    return inputs, forcings
