import torch
import numpy as np
import xarray as xr
from lib.torch import interface
from lib.torch import ForcedStepper


def sel(x):
    return x.isel(x=slice(0, 64))


def mean_abs_dev(x, y):
    return (x-y).apply(np.abs).mean(['x', 'time'])


def _window_errors(model, inputs, forcings, window_size=256, nstarts=10,
                   prognostics=('qt', 'sl')):
    """Yield the MAD computed over randomly selected intervals of a given
    length"""

    prognostics = list(prognostics)

    n = len(inputs.time)
    start_idxs = np.random.choice(n-window_size, nstarts)

    for i in start_idxs:
        times = slice(i, i+window_size)
        ii = inputs.isel(time=times)
        ff = forcings.isel(time=times)
        # run the column model
        progs, _ = interface.column_run(model, ii, ff)
        # compute the MAD
        yield mean_abs_dev(ii[prognostics], progs[prognostics])


def get_test_error(*args, nstarts=10, **kwargs):
    return sum(_window_errors(*args, nstarts=nstarts, **kwargs))/nstarts


def _compute_residual(x, f):
    dt = x.time[1]-x.time[0]
    favg = (f + f.shift(time=1))/2
    return (x.shift(time=-1)-x)/dt - favg


def compute_residuals(inputs, forcings):
    """Compute Q1 and Q2"""
    ds = xr.Dataset({key: _compute_residual(inputs[key], forcings[key])
                     for key in ['qt', 'sl']})
    return ds


def trapezoid_step(x, g, h=.125):
    return x + h * (g + g.shift(time=1))/2


def mse(x, y, dims=('x', 'y', 'time')):
    return ((x-y)**2).mean(dims)


def estimate_source(model, inputs, forcings):
    # estimate source terms
    # need to take a step using the trapezoid rule though
    # see results/8.4-nbd-Q1-Q20-score.ipynb
    xst = inputs.apply(lambda x: trapezoid_step(x, forcings[x.name]) if x.name
                       in ['sl', 'qt'] else x)
    gavg = (forcings + forcings.shift(time=1))/2
    src = interface.rhs(model, xst, gavg)
    return src


def get_src_error(model, inputs, forcings):
    """Compute the errors in the predicted source terms (Q1 and Q2)"""
    resid = compute_residuals(inputs, forcings)
    src = estimate_source(model, inputs, forcings)

    dims = ('x', 'time')
    data_vars = {}

    w = inputs.w

    for f in ['qt', 'sl']:
        sse = mse(src[f], resid[f], dims)
        sst = mse(resid[f], resid[f].mean(dims), dims)
        r2 = 1-(sse * w).sum('z')/(sst*w).sum('z')

        data_vars[f + 'SSE'] = sse
        data_vars[f + 'SS'] = sst
        data_vars[f + 'R2'] = r2

    return xr.Dataset(data_vars)


def main(torch_file, input_path, forcing_path, output):
    model = ForcedStepper.load_from_saved(torch.load(torch_file))
    inputs = xr.open_dataset(input_path).pipe(sel)
    forcings = xr.open_dataset(forcing_path).pipe(sel)
    test_error = get_test_error(model, inputs, forcings)
    src_error = get_src_error(model, inputs, forcings)

    return xr.auto_combine((test_error, src_error))\
             .assign(p=inputs.p, w=inputs.w)\
             .to_netcdf(output)


i = snakemake.input
main(i.model, i.inputs, i.forcings, output=snakemake.output[0])
