import matplotlib.pyplot as plt
import numpy as np

import common
from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap


def rms(x, dim='x'):
    return np.sqrt((x**2).mean(dim))


def corr(*args, dim='x'):
    x, y = [a - a.mean(dim) for a in args]
    sig2_x = (x**2).sum(dim)
    sig2_y = (y**2).sum(dim)
    return (x * y).sum(dim) / np.sqrt(sig2_x * sig2_y)


def get_precipitation():
    net_precip_nn = runs['debias'].data_2d.NPNN

    ds = open_data('training')
    evap = lhf_to_evap(ds.LHF)
    net_precip_truth = ds.Prec - evap

    dsm = runs['micro'].data_2d
    net_precip_control = dsm.Prec - lhf_to_evap(dsm.LHF)

    time = net_precip_nn.time

    return (net_precip_truth.interp(time=time), net_precip_nn,
            net_precip_control)


def get_data():
    truth, nn, micro = get_precipitation()

    corr_nn = corr(truth, nn)
    corr_micro = corr(truth, micro)

    return corr_nn, corr_micro


def plot_pane(ax, plotme, title=''):

    plotme = plotme.sel(time=slice(100, 110))
    im = plotme.plot(x='time', vmin=0, vmax=1, ax=ax, add_colorbar=False)
    plotme.plot.contour(
        x='time',
        levels=[.5],
        colors='k',
        ax=ax, )
    if title:
        ax.set_title(title)
    return im


def plot(data):

    corr_nn, corr_micro = data

    fig, (a, b) = plt.subplots(
        2,
        1,
        sharex=True,
        constrained_layout=True,
        figsize=(common.textwidth / 2, 5))
    im = plot_pane(a, corr_nn, title='NN')
    plot_pane(b, corr_micro, title='Micro + Rad')
    fig.colorbar(im, ax=[a, b])


if __name__ == '__main__':
    plot(get_data())
    plt.savefig("pattern_correlation.png")
