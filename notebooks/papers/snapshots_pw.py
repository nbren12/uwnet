import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid

import common
from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap

output = ["snapshots_pw.png"]


def get_data():

    variables = ['PW']
    times = [105]

    # open NN run
    run = runs['debias']
    nn = run.data_2d.rename({'NPNN': 'net_precip'})

    # open NN run
    run = runs['unstable']
    unstable = run.data_2d.rename({'NPNN': 'net_precip'})

    # open microphysics
    run = runs['micro']
    micro = run.data_2d
    micro['net_precip'] = micro.Prec - lhf_to_evap(micro.LHF)

    # open NGAqua
    ng = open_data('ngaqua_2d')
    ng['net_precip'] = ng.Prec - lhf_to_evap(ng.LHF)
    # make sure the x and y value agree
    ng = ng.assign(x=nn.x, y=nn.y)

    data = [ng, nn, unstable, micro]
    tags = ['NG-Aqua', 'NN', 'Unstable', 'Micro']

    plotme = xr.concat(
        [run[variables].interp(time=times) for run in data], dim=tags)

    return plotme.sel(time=times).squeeze('time')


def plot(plotme):
    fig = plt.figure(1, figsize=(common.textwidth, common.textwidth / 2))

    grid = AxesGrid(
        fig,
        111,  # similar to subplot(144)
        nrows_ncols=(2, 2),
        aspect=True,
        axes_pad=(.4, .3),
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="each",
        cbar_size="7%",
        cbar_pad="2%", )

    count = 0
    abc = 'abcdefghijk'
    for j, run in enumerate(plotme.concat_dim):
        cax = grid.cbar_axes[count]
        ax = grid[count]

        val = plotme.PW[j]
        im = ax.pcolormesh(val.x / 1e6, val.y / 1e6, val.values)
        cax.colorbar(im)

        run = str(run.values)
        ax.set_title(f'{abc[count]}) {run}', loc='left')
        count += 1

    axs = np.reshape(grid, (2, 2))
    common.label_outer_axes(axs, "x (1000 km)", "y (1000 km)")


def main():
    plotme = get_data()
    plot(plotme)
    plt.savefig(output[0])


if __name__ == '__main__':
    main()
