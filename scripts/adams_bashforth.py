import dask.array as da
import numpy as np

from xarray.core.computation import apply_ufunc

from lib.util import xopena


def ab3_dask(x):
    out0 = x[0]
    out1 = x[1] * 1.5 - .5 * x[0]
    out_rest = 23/12*x[2:] - 4/3*x[1:-1] + 5/12*x[0:-2]

    return da.concatenate((out0[np.newaxis,...],
                           out1[np.newaxis,...],
                           out_rest), axis=0)

def main(snakemake):

    # open data
    f = xopena(snakemake.input[0])

    if f.dims[0] !='time':
        raise ValueError('time must be the first dimension')

    # calculate ab3
    out = apply_ufunc(ab3_dask, f, dask_array='allowed')

    # save output
    out.to_netcdf(snakemake.output[0])


try:
    snakemake
except NameError:
    pass
else:
    main(snakemake)
