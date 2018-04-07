"""
In this notebook, I try using the multiple time step objective function. This is given by

$$ \sum_{j=1}^{m} | x^j - \tilde{x}^j|^2,$$
where the approximate time series is defined recursively by $\tilde{x}^0 = x^0$, $\tilde{x}^j=f(\tilde{x}^{j-1}) + (g^i + g^{i+1})\frac{h}{2}$ for $j>1$. Here, $f$ will be approximated using a neural network.

"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import xarray as xr
import os

import torch
from ..torch import column_run
# from lib.evaluation.single_column import forced_step, runsteps, step


def plot_soln(x, fig=None, q_levs=None, dims=['time', 'p'], figsize=None):

    if fig is None:
        fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2,
                  width_ratios=(.97, .03),
                  height_ratios=(1, 1, .5),
                  wspace=.01)

    if not q_levs:
        q_levs = np.arange(11) * 2.5

    t_levs = np.arange(12) * 5 + 275

    args = [x[dim].values for dim in dims]

    x = x.squeeze()

    ax = plt.subplot(gs[0,0])

    axs = [ ax,
        plt.subplot(gs[1,0], sharex=ax, sharey=ax),
        plt.subplot(gs[2,0], sharex=ax)
    ]



    t_im = axs[0].contourf(*args, x.sl.T, levels=t_levs, extend='both')
    # q_im = axs[1].contourf(*args, x.qt.T, levels=qt_levs, extend='both')
    q_im = axs[1].contourf(*args, x.qt.T, levels=q_levs, extend='both')

    plt.colorbar(t_im, cax=plt.subplot(gs[0, 1]))
    plt.colorbar(q_im, cax=plt.subplot(gs[1, 1]))


    axs[0].set_ylabel('p (mb)')
    axs[1].set_ylabel('p (mb)')
    axs[2].plot(x.prec.time, x.prec)

    axs[0].text(.02, .85, '$s_l$ (K)', bbox=dict(color='white'),
                transform=axs[0].transAxes)
    axs[1].text(.02, .85, '$q_T$ (g/kg)', bbox=dict(color='white'),
                transform=axs[1].transAxes)
    axs[2].text(.02, .8, 'P (mm/day)', bbox=dict(color='white'),
                transform=axs[2].transAxes)

    axs[0].invert_yaxis()

    time = x.time
    begin = float(time[0])
    end = float(time[-1])
    axs[0].set_xlim([begin, end])

    return axs


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

    mu = forcings.mean('time') + forcings*0
    plot_column_run(p, model, inputs, mu)
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
