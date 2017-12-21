import numpy as np
import dask.array as da

from xarray.core.computation import apply_ufunc

from lib.util import xopena


def diff(x):
    dx = (x[1:] - x[:-1])
    return da.concatenate((dx[0:1], dx), axis=0)


def main(snakemake):

    # open data
    f = xopena(snakemake.input.forcing)
    x = xopena(snakemake.input.data)

    if x.dims[0] != 'time':
        raise ValueError('time must be the first dimension')

    if x.time.attrs['units'] != 'd':
        raise ValueError('units of time must be in d')

    h = float(f.time[1] - f.time[0])

    # calculate ab3
    dt = apply_ufunc(diff, x, dask_array='allowed')
    # subtract forcing terms
    out = dt/h - f

    out.name = 'f_' + x.name
    out.attrs['units'] = x.units + '/d'
    out.attrs['long_name'] = f'apparent source of {x.name}'

    # save output
    out.to_netcdf(snakemake.output[0])


try:
    snakemake
except NameError:
    pass
else:
    main(snakemake)
