"""
In this notebook, I try using the multiple time step objective function. This is given by

$$ \sum_{j=1}^{m} | x^j - \tilde{x}^j|^2,$$
where the approximate time series is defined recursively by $\tilde{x}^0 = x^0$, $\tilde{x}^j=f(\tilde{x}^{j-1}) + (g^i + g^{i+1})\frac{h}{2}$ for $j>1$. Here, $f$ will be approximated using a neural network.

"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.externals import joblib
import os

import torch
# from lib.evaluation.single_column import forced_step, runsteps, step
from lib.models.torch import wrap


def plot_soln(x):

    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharey=True)
    qt_levs = np.arange(11) * 2.5

    t_levs = np.arange(12) * 5 + 275

    args = (x.time.values, x.p.values)

    x = x.squeeze()

    t_im = axs[0].contourf(*args, x.sl.T, levels=t_levs, extend='both')
    q_im = axs[1].contourf(*args, x.qt.T, levels=qt_levs, extend='both')

    plt.colorbar(t_im, ax=axs[0])
    plt.colorbar(q_im, ax=axs[1])

    axs[0].set_ylabel('sl')
    axs[1].set_ylabel('qt')

    axs[0].invert_yaxis()


def dataset_to_dict(dataset):
    out = {}
    for key in dataset.data_vars:
        x = dataset[key]
        if 'z' not in x.dims:
            x = x.expand_dims('z')
        x = x.expand_dims('batch')
        out[key] = x
    return out


def column_run(model, prognostic, forcing):
    prog = dataset_to_dict(prognostic)
    forcing = dataset_to_dict(forcing)

    prog.pop('p')
    prog.pop('w')

    y = wrap(model)({'prognostic': prog, 'forcing': forcing})

    coords = prog['sl'].coords
    dims = ['time', 'batch', 'z']

    return {
        key: xr.DataArray(y['prognostic']['sl'], coords=coords, dims=dims)
        for key in y['prognostic']
    }


def plot_column_run(p, *args):
    y = xr.Dataset(column_run(*args)).assign(p=p)
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

    plot_soln(inputs)
    path = os.path.join(output_dir, "truth.png")
    plt.savefig(path)

    return {
        'forced': "forced.png",
        'unforced': "unforced.png",
        'truth': 'truth.png'
    }
