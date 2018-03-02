import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from .common import hide_xlabels


def _plot_bias_sig(loc, ax=None, colors=('k', 'b', 'g')):

    dims = ['x', 'time']
    mu = loc.mean(dims)
    sig = loc.std(dims)
    bias = mu - mu.sel(model='Truth')

    ds = xr.Dataset({'mu': mu, 'sig': sig, 'bias': bias})

    lines = {}
    for c, (model, val) in zip(colors, ds.groupby('model')):

        # show bias only if not plotting the Truth
        if model == 'Truth':
            b = val.bias * np.NaN
        else:
            b = val.bias
        b = b.squeeze()

        ax.plot(val.sig.squeeze(), val.p, c=c, ls='--')
        l, = ax.plot(b, b.p, c=c)
        # store these later for the legend
        lines[model] = l
    ax.set_ylim([1000, 0])
    return lines


def plot_profiles(loc, axs=None):

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(4, 3), dpi=100, sharey=True,
                                gridspec_kw=dict(wspace=0.04))

    lines = _plot_bias_sig(loc.qt, ax=axs[0])
    lines = _plot_bias_sig(loc.sl, ax=axs[1])
    axs[1].set_xlim([-2, 4])

    # labels and legends
    axs[0].set_title(r'$q_T$')
    axs[0].set_xlabel('(g/kg)')
    # axs[0].set_ylabel('p (hPa)')

    axs[1].set_title(r'$s_L$')
    axs[1].set_xlabel('K')
    axs[0].legend(lines.values(), lines.keys(), bbox_to_anchor=(1.0, .9))

    return axs


def _plot_pres_vs_lat(bias, ax, levels=np.arange(-5, 6)*.5, title=None):
    im = ax.contourf(bias.y/1000, bias.p, bias, levels, cmap='bwr',
                     extend='both')

    plt.colorbar(im, pad=.01, ax=ax)
    ax.set_xlabel('y (1000 km)')
    if title:
        ax.set_title(title)


def plot_bias_pres_vs_lat(ds_test, axs):
    """Plot bias of a dataset on pressure vs latitude"""
    bias = (ds_test.sel(model='Neural Network').mean(['x', 'time'])
            - ds_test.sel(model='Truth').mean(['x', 'time']))

    _plot_pres_vs_lat(bias.qt, axs[0], title="Humidity bias (g/kg)")
    _plot_pres_vs_lat(bias.sl, axs[1], title="Temperature bias (K)")


def plot(ds_test, width=5.5):
    gridspec_kw = dict(
        width_ratios=(.7, .3)
    )

    fig, axs = plt.subplots(2, 2, figsize=(width, width), sharey=True,
                            gridspec_kw=gridspec_kw)

    plot_profiles(ds_test.isel(y=8), axs=axs[:, 1])
    plot_bias_pres_vs_lat(ds_test, axs=axs[:, 0])

    hide_xlabels(axs[0, 0])
    plt.tight_layout()
