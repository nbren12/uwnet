import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import common
from src.data import runs, training_data
from uwnet.thermo import vorticity


RUNS = ['debias', 'unstable', 'micro']
time_range = slice(100.625, 110.625)


def rms(x, dim=None):
    if dim is None:
        dim = ['x', 'y']
    return np.sqrt((x**2).mean(dim))


def global_mass_weighted_rms(ds, reference='truth'):
    truth = ds.sel(concat_dim=reference)
    preds = ds.sel(concat_dim=ds.concat_dim != reference)

    M = ds.layer_mass.sum('z')
    squares = (truth - preds)**2 * ds.layer_mass / M
    sum_squares = squares.sum('z', skipna=False).mean(['x', 'y'])
    return np.sqrt(sum_squares)


# @common.cache
def get_data():

    ds = get_merged_data(['U', 'V', 'SLI', 'QT'],
                         RUNS)
    # compute vorticity
    ds['VORT'] = vorticity(ds.U, ds.V)

    regions = common.get_regions(ds.y)
    rms = ds.groupby(regions).apply(global_mass_weighted_rms)
    return rms


def get_merged_data(variables, run_names):
    truth = xr.open_dataset(training_data).sel(time=time_range)
    data = {'truth': truth[variables]}

    for run in run_names:
        ds = runs[run].data_3d[variables].load()
        data[run] = ds.interp(time=truth.time)

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
            markevery=15)
        lines.append(l)
        labels.append(f'{common.run_labels[run]} in {region}')

    ax.set_title(title, loc='left')

    return lines, labels


def plot_old(ds):

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(common.textwidth, 4))
    
    a = axs[0,0]
    b = axs[0,1]
    c = axs[1,0]
    
    plot_rms_runs_regions_times(ds.QT, a, title="a) QT (g/kg)")
    plot_rms_runs_regions_times(ds.SLI, b, title="b) SLI (K)")
    lines, labels = plot_rms_runs_regions_times(
        ds.VORT * 1e6, c, title=r"c) vertical vorticity (10^-6 s^-1)")

    a.set_xlim([101, 108.5])
    [ax.set_ylim(bottom=0.0) for ax in (a, b, c)]
    common.label_outer_axes(np.array([[a], [b], [c]]), "time (day)", "")

    common.despine_axes([a, b, c])
    
    axs[1,1].axis('off')

    fig.legend(
        lines,
        labels,
        loc="lower right",
        bbox_to_anchor=(.9, 0.1),
        frameon=False)

    
def hide_spines(ax, spines):
    for spine in spines:
        spine = ax.spines[spine]
        spine.set_visible(False)
        
def plot(df):

#     df = df.sel(time=slice(None, 108.5))
    df['time'] = df.time - df.time[0]
    df['VORT'] *= 1e6
    
    fig, axs = plt.subplots(3, 3, figsize=(common.textwidth, common.textwidth/1.3), constrained_layout=True)
    letters = 'abcdefghijklm'
    count = 0
    
    nregion = 3
    nvars = 3
    nrun = len(RUNS)

    for i in range(nvars):
        for j in range(nregion):
            for k in range(nrun):

                ax = axs[i,j]
                run = str(df.concat_dim[k].values)
                region = str(df.y[j].values)

                field = ['SLI', 'QT', 'VORT'][i]
                unit = ['K', 'g/kg', '10^-6 s^-1'][i]
    #             ylim = [4.5, 3, 50][i]


                data = df[field].sel(y=region, concat_dim=run)
                label = common.run_labels[run]

                ax.plot(data.time, data, label=label)
    #             ax.set_ylim(top=ylim)
                ax.xaxis.set_major_locator(plt.MaxNLocator(4))

                if j == 0:
                    ax.set_ylabel(f'{unit}')
                hide_spines(ax, ['top',  'right'])


                if i == 2:
                    ax.set_xlabel('lead time (days)')

                letter = letters[count]
        #         ax.text(.1, .95, f'{letter})', transform=ax.transAxes)
                ax.set_title(f'{letter}) {field} {region}', loc='left')

            count += 1

    axs[-1,0].legend(frameon=False)

if __name__ == '__main__':
    import sys
    output = sys.argv[1]
    
    df = get_data()
    plot(df)
    plt.savefig(output)