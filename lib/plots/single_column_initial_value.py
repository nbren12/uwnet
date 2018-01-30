"""
In this notebook, I try using the multiple time step objective function. This is given by

$$ \sum_{j=1}^{m} | x^j - \tilde{x}^j|^2,$$
where the approximate time series is defined recursively by $\tilde{x}^0 = x^0$, $\tilde{x}^j=f(\tilde{x}^{j-1}) + (g^i + g^{i+1})\frac{h}{2}$ for $j>1$. Here, $f$ will be approximated using a neural network.

"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import xarray as xr
from sklearn.externals import joblib
import os

import torch
from torch.autograd import Variable
# from lib.evaluation.single_column import forced_step, runsteps, step


def plot_soln(x):

    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(3, 2,
                  width_ratios=(.97, .03),
                  height_ratios=(1, 1, .5),
                  wspace=.01)

    qt_levs = np.arange(11) * 2.5

    t_levs = np.arange(12) * 5 + 275

    args = (x.time.values, x.p.values)

    x = x.squeeze()

    ax = plt.subplot(gs[0,0])

    axs = [ ax,
        plt.subplot(gs[1,0], sharex=ax, sharey=ax),
        plt.subplot(gs[2,0], sharex=ax)
    ]



    t_im = axs[0].contourf(*args, x.sl.T, levels=t_levs, extend='both')
    q_im = axs[1].contourf(*args, x.qt.T, levels=qt_levs, extend='both')

    plt.colorbar(t_im, cax=plt.subplot(gs[0, 1]))
    plt.colorbar(q_im, cax=plt.subplot(gs[1, 1]))

    axs[0].set_ylabel('sl')
    axs[1].set_ylabel('qt')
    axs[2].plot(x.prec.time, x.prec)

    axs[0].invert_yaxis()
    axs[0].set_xlim([100, 180])


def dataset_to_dict(dataset):
    out = {}
    for key in dataset.data_vars:
        x = dataset[key]
        if 'z' not in x.dims:
            x = x.expand_dims('z')
        x = x.expand_dims('batch')

        transpose_dims = [dim for dim in ['time', 'batch', 'z']
                          if dim in x.dims]
        x = x.transpose(*transpose_dims)
        x = Variable(torch.FloatTensor(x.values))
        out[key] = x
    return out


def column_run(model, prognostic, forcing):
    prog = dataset_to_dict(prognostic)
    forcing = dataset_to_dict(forcing)

    prog.pop('p')
    w = prog.pop('w')

    input_data = {
        'prognostic': prog,
        'forcing': forcing,
        'constant': {'w': w}
    }

    y = model(input_data)

    coords = {'z': prognostic['z'], 'time': prognostic['time']}
    dims = ['time', 'batch', 'z']

    progs = {
        key: xr.DataArray(y['prognostic'][key].data.numpy(), coords=coords, dims=dims)
        for key in y['prognostic']
    }

    prec = xr.DataArray(y['diagnostic']['Prec'].data.numpy().ravel(),
                        coords={'time': prognostic.time[:-1]},
                        dims=['time'])

    return progs, prec


def plot_column_run(p, *args):
    y, prec = column_run(*args)
    y = xr.Dataset(y)
    y['prec'] = prec
    y['p'] = p
    plot_soln(y)


def main(inputs, forcings, torch_file, output_dir):
    def mysel(x):
        return x.isel(x=0, y=8).transpose('time', 'z')

    model = torch.load(torch_file)
    inputs = xr.open_dataset(inputs).pipe(mysel)
    forcings = xr.open_dataset(forcings).pipe(mysel)
    p = inputs.p

    plot_column_run(p, model, inputs, 0*forcings)
    unforced_path = os.path.join(output_dir, "unforced.png")
    plt.savefig(unforced_path)

    plot_column_run(p, model, inputs, forcings)
    forced_path = os.path.join(output_dir, "forced.png")
    plt.savefig(forced_path)

    inputs['prec'] = forcings.Prec
    plot_soln(inputs)
    path = os.path.join(output_dir, "truth.png")
    plt.savefig(path)

    return {
        'forced': "forced.png",
        'unforced': "unforced.png",
        'truth': 'truth.png'
    }
