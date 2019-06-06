import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import common
from src.data import open_data, runs

avg_days = [105, 110]


def avg(nn):
    return nn.sortby('time').sel(time=slice(*avg_days)).mean(['x', 'time'])


def get_data():

    ds = open_data('training')

    run = runs['debias']
    nn = run.data_3d

    common_variables = list(set(nn.data_vars) & set(ds.data_vars))
    plotme = xr.concat(
        [avg(nn[common_variables]),
         avg(ds[common_variables])],
        dim=['NN', 'NG-Aqua'])

    length_domain = 160e3 * len(ds.x)
    plotme['stream_function'] = (
        plotme.V * ds.layer_mass[0]).cumsum('z') * length_domain

    plotme = plotme.assign(p=plotme.p[0]).swap_dims({'z': 'p'})

    return plotme


def compare_climate(da, ax, spacing=None, reference=None, title=''):

    da = da.assign_coords(y=da.y/1e6)
    da.y.attrs['units'] = '1000 km'

    nn = da.sel(concat_dim='NN')
    ng = da.sel(concat_dim='NG-Aqua')

    bias = nn - ng

    bias.plot.contourf(ax=ax, yincrease=False, levels=11, add_labels=False)

    if spacing:
        a = ng.min() // spacing
        levels = np.arange(a, (a + 20)) * spacing
    else:
        levels = 11


    kwargs = {}
    if reference is not None:
        i = np.argmin(np.abs(np.array(levels) - reference))
        linewidths = [1.0] * len(levels)
        linewidths[i] = 1.5
        kwargs['linewidths'] = linewidths

    cs = ng.plot.contour(
        ax=ax, yincrease=False, colors='k', levels=levels, add_labels=False,
        **kwargs)

    if reference is not None:
        plt.clabel(cs, [levels[i]], fmt="%.0f", fontsize=common.clabel_size)

    ax.set_title(title + f'\tspacing = {spacing}', loc='left')


def plot(plotme):
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(common.textwidth, common.textwidth / 2),
        sharey=True,
        sharex=True)

    (a, b), (c, d) = axs

    compare_climate(plotme.QT, a, spacing=4, reference=4, title=r'a) $q_T$ (g/kg)')
    compare_climate(plotme.SLI, b, spacing=10, reference=300, title=r'b) $s_L$ (K)')
    compare_climate(plotme.U, c, spacing=10, reference=0, title='c) $u$ (m/s)')
    compare_climate(plotme.stream_function / 1e9, d, spacing=20, reference=0,
                    title=r'd) $\psi$ (Tg/s)')

    common.label_outer_axes(axs, "y (1000 km)", "p (mbar)")


if __name__ == '__main__':
    data = get_data()
    plot(data)
    plt.savefig("bias.pdf")
