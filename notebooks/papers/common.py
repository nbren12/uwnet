import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors
import joblib

plt.style.use('tableau-colorblind10')

memory = joblib.Memory(location='cache', verbose=1)
cache = memory.cache

run_labels = {'debias': 'NN-Lower', 'unstable': 'NN-All', 'micro': 'Base'}

ignored_input_levels = {'QT': 442, 'SLI': 267}

def get_model(name):
    if name == 'NN-Lower':
        return torch.load("../../nn/NNLower/4.pkl")
    if name == 'NN-All':
        return torch.load("../../nn/NNAll/5.pkl")

def setup_matplotlib():

    params = {
        'axes.labelsize': 8,
        'font.size': 10,
        'text.usetex': False,
        'legend.fontsize': 'small',
        'figure.figsize': [4.5, 4.5],
        'savefig.dpi': 150
    }

    plt.rcParams.update(params)
    plt.rc('axes', titlesize='medium')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rc('lines', linewidth=1.0, markersize=3)


def data_array_dict_to_dataset(d, dim='keys'):
    idx = pd.Index(d.keys(), name=dim)
    return xr.concat(d.values(), dim=idx)


def to_pressure_dim(ds):
    from src.data import open_data
    return ds.assign_coords(p=open_data('pressure')).swap_dims({'z': 'p'})


def get_vmax(val):
    a = val.min()
    b = val.max()

    return max([abs(a), abs(b)])


def label_outer_axes(axs, xlabel, ylabel):
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)

    for ax in axs[-1, :]:
        ax.set_xlabel(xlabel)


def despine_axes(axs, spines=('right', 'top')):

    for ax in axs:
        for spine in spines:
            ax.spines[spine].set_visible(False)


def get_regions(y):
    """Get the tropics subtropics and extratropics mask

    This is especially useful for groupby operations.
    """
    tropics_bndy = .25
    subtropics_north_bndy = .50
    dx = 160e3

    percent = 2 * y / (y.max() + dx) - 1

    subtropics = ((np.abs(percent) > tropics_bndy) &
                  (np.abs(percent) <= subtropics_north_bndy))

    tropics = np.abs(percent) <= tropics_bndy
    return xr.where(tropics, 'Tropics',
                    xr.where(subtropics, 'Subtropics', 'Extratropics'))


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def diverging_levels(m, s):
    n = m // s + 1
    return np.r_[-n:n+1] * s
    

textwidth = 6.5
onecolumn = textwidth/2 - .5

setup_matplotlib()
clabel_size = 'smaller'
