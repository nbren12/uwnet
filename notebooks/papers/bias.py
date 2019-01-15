import xarray as xr

from src.data import open_data, runs
import matplotlib.pyplot as plt

avg_days = [105, 110]


def avg(nn):
    return nn.sel(time=slice(*avg_days)).mean(['x', 'time'])


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
        plotme.V * ds.layer_mass).cumsum('z') * length_domain

    plotme = plotme.assign(p=plotme.p[0]).swap_dims({'z': 'p'})

    return plotme


def compare_climate(da, ax):

    nn = da.sel(concat_dim='NN')
    ng = da.sel(concat_dim='NG-Aqua')

    bias = nn - ng

    bias.plot.contourf(ax=ax, yincrease=False, levels=11)
    cs = ng.plot.contour(ax=ax, yincrease=False, colors='k', levels=11)
    plt.clabel(cs)


def plot(plotme):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharey=True, sharex=True)

    (a, b), (c, d) = axs

    compare_climate(plotme.QT, a)
    compare_climate(plotme.SLI, b)
    compare_climate(plotme.U, c)
    compare_climate(plotme.stream_function / 1e9, d)


if __name__ == '__main__':
    plot(get_data())
    plt.savefig("bias.png")
