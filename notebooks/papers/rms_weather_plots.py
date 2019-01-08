import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from src.data import training_data, runs
from common import data_array_dict_to_dataset
import seaborn as sns


def rms(x, dim=None):
    if dim is None:
        dim = ['x', 'y']
    return np.sqrt((x**2).mean(dim))


def regional_average_error(out_eror):
    tropics_bndy = .1
    dx = 160e3

    y = out_eror.y
    percent = 2 * y / (y.max() + dx) - 1

    subtropics = (np.abs(percent) > tropics_bndy) & (np.abs(percent) <= .5)
    tropics = np.abs(percent) <= tropics_bndy
    region = xr.where(tropics, 'Tropics',
                      xr.where(subtropics, 'Subtropics', 'Extratropics'))

    out_eror['region'] = region
    avg_ss = out_eror.groupby('region').apply(
        lambda x: (x**2).mean('y')).compute()

    return avg_ss.to_dataframe().reset_index()


def get_data(field='QT', z=10, tlim=slice(100, 110), avg_dims='x'):
    truth = xr.open_dataset(training_data).isel(step=0)

    out_eror = {}

    for run in runs:
        pred = runs[run].data_3d[field].sel(time=tlim).isel(z=z)
        target = truth[field].isel(z=z)
        out_eror[run] = rms(pred - target, dim=avg_dims)

    out_eror = data_array_dict_to_dataset(out_eror)

    return regional_average_error(out_eror)


def plot(df):
    df['RMS'] = np.sqrt(df.QT)
    sns.FacetGrid(df, hue="keys", col="region")\
       .map(plt.plot, "time", "RMS")\
       .add_legend()


df = get_data()
plot(df)
