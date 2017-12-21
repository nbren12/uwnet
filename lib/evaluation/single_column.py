import numpy as np
import xarray as xr
import torch
from toolz import curry
from .models.torch_models import predict

def runsteps(step, x, n):
    """Perform n steps using"""
    out = np.zeros((n, x.shape[0]), dtype=x.dtype)
    out[0] = x

    for i in range(n):
        x = step(x, i)
        out[i] = x

    return out


@curry
def step(net, x, n):
    return predict(net, x[None,:])[0,:]


@curry
def forced_step(net, g, x, n):

    if n < g.shape[0]-1:
        f = (g[n] + g[n+1])/2
    else:
        f = g[n]

    return predict(net, x[None,:])[0,:] + f * 3/24


@curry
def xr_runsteps(stepper, inputs, forcings, weights,
                n=None, **kwargs):
    """Run the single column test with xarray input and output

    Parameters
    ----------
    stepper
        callable that takes data in stacked format
    prepare_data
        callable that processed data into data_dict format
    inputs : xr.Dataset
        prognostic variables to be passed prepare_data
    outputs : xr.Dataset
        forcing variables to be passed prepare_data
    weights: xr.DataArray
        layer masses
    n : int
        optional number of steps to perform
    **kwargs:
        passed to prepare_data

    Returns
    -------
    xarray dataset with same length as X

    """
    from lib.preprocess import prepare_data, unstacked_data


    data = prepare_data(inputs, forcings, weights, **kwargs)
    X = data['X']
    G = data['G']

    if n is None:
        n = X.shape[0]

    x_out = runsteps(forced_step(stepper, G), X[0], n)

    state = unstacked_data(x_out)
    var_dict = {key: (['time', 'z'], state[key]) for key in state}
    return xr.Dataset(var_dict, coords=inputs.coords)


def lagged_predictions(stepper, inputs, forcings, weights, n=30, **kwargs):
    """Compute the lagged predictions given inputs and forcings

    Parameters
    ----------
    stepper : nn.Module
        torch module for time stepper
    *args :
        same input arguments as lib.preprocess.prepare_data
    n : int, optional
        number of lags to perform predictions for. default: 30

    Returns
    -------
    lagged_predictions : xr.Dataset
        dataset with dimensions (lag, time, z)
    """
    from lib.preprocess import prepare_data, unstacked_data


    data = prepare_data(inputs, forcings, weights, **kwargs)
    X = data['X']
    G = data['G']

    nt, nf = X.shape

    # shape has prediction lag, time, feature
    output = np.ones((n, nt, nf)) * np.nan

    # run prediction for nsteps from each point
    for i in range(nt):
        x_out = runsteps(forced_step(stepper, G), X[i], min(n, nt-i))
        for j in range(x_out.shape[0]):
            output[j, i+j, :] = x_out[j]

    state = unstacked_data(output)
    var_dict = {key: (['lag', 'time', 'z'], state[key]) for key in state}
    return xr.Dataset(var_dict, coords=inputs.coords)
