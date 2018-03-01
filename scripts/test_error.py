import torch
import numpy as np
import xarray as xr
from lib.plots.single_column_initial_value import column_run


def sel(x):
    return x.isel(x=slice(0, 64))


def mean_abs_dev(x, y):
    return (x-y).apply(np.abs).mean(['x', 'time'])


def get_test_error(model, inputs, forcings, prognostics=('qt', 'sl')):
    prognostics = list(prognostics)
    i = inputs.stack(batch=['x', 'y'])
    f = forcings.stack(batch=['x', 'y'])
    progs, prec = column_run(model, i, f)
    data = xr.Dataset(progs)\
             .unstack("batch")
    return mean_abs_dev(inputs[prognostics], data[prognostics])


def main(torch_file, input_path, forcing_path, output):
    model = torch.load(torch_file)
    inputs = xr.open_dataset(input_path).pipe(sel)
    forcings = xr.open_dataset(forcing_path).pipe(sel)
    return get_test_error(model, inputs, forcings).to_netcdf(output)


i = snakemake.input
main(i.model, i.inputs, i.forcings, output=snakemake.output[0])
