import matplotlib.pyplot as plt
import numpy as np
from toolz import valmap
from toolz.curried import get

import xarray as xr
from xnoah import integrate

from .common import despine


def compute_errors(metric, ds, **kwargs):

    mad = {}
    truth = ds.sel(model='Truth')

    for key in ['Neural Network', 'CAM']:
        if key in ds.model.values.tolist():
            mad[key] = metric(
                truth, ds.sel(model=key), **kwargs).sortby('time')

    mad['Mean'] = metric(truth, truth.mean(['x', 'time']),
                         **kwargs).sortby('time')
    mad['Persistence'] = metric(
        truth, truth.isel(time=0), **kwargs).sortby('time')

    return mad


def mean_squared_error(truth, pred, dims=('x', )):
    return ((truth - pred).fillna(0.0)**2).mean(dims)


def mean_absolute_dev(truth, pred, dims=('x', )):
    return (truth - pred).fillna(0.0).apply(np.abs).mean(dims)


def plot_mses(mses, axs=None, label='', n_days=40, **kwargs):
    if axs is None:
        fig, axs = plt.subplots(
            2, 2, sharey=True, sharex=True, dpi=100, figsize=(6, 3.5))

    keys = mses.keys()

    for ax, key in zip(axs.flat, keys):
        val = mses[key]
        im = ax.contourf(val.time, val.p, val.T, **kwargs)

        ax.text(
            .05, .8, key, transform=ax.transAxes, color='white', fontsize=13)

    axs[0, 0].invert_yaxis()
    axs[0, 0].set_ylabel('p (hPa)')
    axs[1, 0].set_ylabel('p (hPa)')

    for ax in axs[1, :]:
        ax.set_xlabel('days')

    plt.subplots_adjust(wspace=.02, hspace=.02)
    cb = plt.colorbar(im, ax=axs, pad=.01)
    cb.set_label(label)

    axs[0, 0].set_xlim([val.time.min(), val.time.min() + n_days])

    return axs, cb


def _adjust_column_error_plots(axQ, axT):
    axs = [axQ, axT]
    for ax in axs:
        despine(ax)
        ax.set_xlabel('time (days)')
        ax.set_xlim([99, 180])

    axQ.set_ylim([0, 1.2])
    axQ.set_ylabel(r'MAD (g/kg)')
    axQ.set_title("a) $q_T$ error", loc="left")

    axT.set_title("b) $s_L$ error", loc="left")
    axT.set_ylim([0, 3.3])
    axT.set_ylabel(r'MAD (K)')
    axT.set_xlim([100, 180])

    # legend
    axQ.legend(ncol=2, columnspacing=0.30,
               bbox_to_anchor=(0.0, .80), loc="lower left")


def plot_column_error(ds):
    # ds_no_cam = ds.drop('CAM', dim='model').sel(y=slice(-400e3, 400e3))

    mad = compute_errors(mean_absolute_dev, ds, dims=['x'])
    mad_sl = valmap(get('sl'), mad)
    mad_qt = valmap(get('qt'), mad)

    mass_mad_sl = xr.Dataset(
        {k: -integrate(v, 'p') / 1015
         for k, v in mad_sl.items()}).to_dataframe()
    mass_mad_qt = xr.Dataset(
        {k: -integrate(v, 'p') / 1015
         for k, v in mad_qt.items()}).to_dataframe()

    with plt.rc_context({
            'axes.prop_cycle': plt.cycler('color', 'kbgy'),
            'lines.linewidth': 1.0,
    }):

        fig, (axQ, axT) = plt.subplots(
            2, 1, figsize=(3, 5), dpi=100, sharex=True)

        mass_mad_sl.plot(ax=axT, legend=False)
        mass_mad_qt.plot(ax=axQ, legend=False)
        _adjust_column_error_plots(axQ, axT)
        plt.tight_layout()

        return axQ, axT
