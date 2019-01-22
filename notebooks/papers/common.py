import pandas as pd
import xarray as xr

from sklearn.externals import joblib
import matplotlib.pyplot as plt

plt.style.use('tableau-colorblind10')

memory = joblib.Memory(location='cache', verbose=1)
cache = memory.cache


def setup_matplotlib():

    params = {
        'axes.labelsize': 8,
        'font.size': 10,
        'text.usetex': False,
        'figure.figsize': [4.5, 4.5],
        'savefig.dpi': 150
    }

    plt.rcParams.update(params)
    plt.rc('axes', titlesize='medium')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')


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


textwidth = 6.5

setup_matplotlib()
