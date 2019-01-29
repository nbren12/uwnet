import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import common
from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap

outputs = ["hov_mean_prec.png", "hov_mean_pw.png"]


def get_data():

    variables = ['PW', 'net_precip']
    times = slice(100, 120)

    # open NN run
    run = runs['debias']
    nn = run.data_2d.rename({'NPNN': 'net_precip'})

    # open microphysics
    run = runs['micro']
    micro = run.data_2d
    micro['net_precip'] = micro.Prec - lhf_to_evap(micro.LHF)

    # open NGAqua
    ng = open_data('ngaqua_2d')
    ng['net_precip'] = ng.Prec - lhf_to_evap(ng.LHF)
    # make sure the x and y value agree
    ng = ng.assign(x=nn.x, y=nn.y)

    plotme = xr.concat(
        [ng[variables].interp(time=nn.time), nn[variables], micro[variables]],
        dim=['NG-Aqua', 'NN', 'Micro'])

    return plotme.sel(time=times).mean('x')


def iterate_over_dim(data, dim):
    for i in range(len(data[dim])):
        yield str(data[dim][i].values), data.isel(**{dim: i})


def plot_row(groupby, letter=slice(None), label='', **kwargs):
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(common.onecolumn, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True)

    abc = 'abcdefg' [letter]
    for a, ax, val in zip(abc, axs, groupby):
        g, val = val
        im = ax.contourf(val.time, val.y / 1e6, val.squeeze().T, **kwargs)
        ax.set_title(f'{a}) {g}', loc='left')

        ax.set_ylabel('y (1000 km)')

    axs[-1].set_xlabel('time (day)')
    cb = plt.colorbar(im, ax=axs, pad=-.01, aspect=50, orientation='horizontal')
    cb.set_label(label)


def main():
    plotme = get_data()

    plot_row(
        iterate_over_dim(plotme.PW, 'concat_dim'),
        label='PW (mm)',
        cbar_kwargs=dict(pad=.02),
        levels=np.r_[5:55:5],
        add_labels=False,
        letter=slice(0, None, 2))
    plt.savefig("hov_mean_pw.pdf")

    plot_row(
        iterate_over_dim(plotme.net_precip, 'concat_dim'),
        label='Column Drying (mm/day)',
        cbar_kwargs=dict(pad=.02),
        levels=np.r_[-10:11:2],
        cmap='RdBu_r',
        extend='both',
        letter=slice(1, None, 2))
    plt.savefig("hov_mean_prec.pdf")


if __name__ == '__main__':
    main()
