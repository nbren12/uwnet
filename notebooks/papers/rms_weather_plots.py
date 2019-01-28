import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import common
from src.data import runs, training_data
from uwnet.thermo import vorticity


def rms(x, dim=None):
    if dim is None:
        dim = ['x', 'y']
    return np.sqrt((x**2).mean(dim))


def global_mass_weighted_rms(ds, reference='truth'):
    truth = ds.sel(concat_dim=reference)
    preds = ds.sel(concat_dim=ds.concat_dim != reference)

    M = ds.layer_mass.sum('z')
    squares = (truth - preds)**2 * ds.layer_mass / M
    sum_squares = squares.sum('z').mean(['x', 'y'])
    return np.sqrt(sum_squares)


@common.cache
def get_data():

    ds = get_merged_data(['U', 'V', 'SLI', 'QT'],
                         ['debias', 'unstable', 'micro'])
    # compute vorticity
    ds['VORT'] = vorticity(ds.U, ds.V)

    regions = common.get_regions(ds.y)
    rms = ds.groupby(regions).apply(global_mass_weighted_rms)
    return rms


def get_merged_data(variables, run_names):
    truth = xr.open_dataset(training_data).isel(step=0)\
                                          .drop('step')\
                                          .sel(time=slice(100, 110))
    data = {'truth': truth[variables]}

    for run in run_names:
        data[run] = runs[run].data_3d[variables].load().interp(time=truth.time)

    ds = xr.concat(data.values(), dim=list(data.keys()))
    ds['layer_mass'] = truth.layer_mass
    return ds


def plot_rms_runs_regions_times(da, ax, title=""):
    keys = da.y.values.tolist()
    colors = dict(zip(keys, ['k', 'b', 'g']))

    keys = da.concat_dim.values.tolist()
    marker = dict(zip(keys, ['', '^', 'o']))
    marker = dict(zip(keys, ['-', ':', '--']))

    lines = []
    labels = []
    for (run, region), val in da.stack(key=['concat_dim', 'y']).groupby('key'):
        l, = ax.plot(
            val.time,
            val,
            ls=marker[run],
            color=colors[region],
            markevery=15,
            label=f'{run} {region}')
        lines.append(l)
        labels.append(f'{run} in {region}')

    ax.set_title(title, loc='left')

    return lines, labels


def plot(ds):

    fig, (a, b, c) = plt.subplots(3, 1, sharex=True, figsize=(5, 6))
    plot_rms_runs_regions_times(ds.QT, a, title="a) QT (g/kg)")
    plot_rms_runs_regions_times(ds.SLI, b, title="b) SLI (K)")
    lines, labels = plot_rms_runs_regions_times(
        ds.VORT * 1e6, c, title=r"c) vertical vorticity (10^-6 s^-1)")

    a.set_xlim([101, 108.5])
    [ax.set_ylim(bottom=0.0) for ax in (a, b, c)]
    common.label_outer_axes(np.array([[a], [b], [c]]), "time (day)", "")

    common.despine_axes([a, b, c])

    plt.subplots_adjust(right=.5)
    fig.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.5, 0.90),
        frameon=False)


if __name__ == '__main__':
    df = get_data()
    plot(df)
