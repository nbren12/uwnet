import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce

from src.data import training_data, runs
from common import data_array_dict_to_dataset, cache
import seaborn as sns


def rms(x, dim=None):
    if dim is None:
        dim = ['x', 'y']
    return np.sqrt((x**2).mean(dim))


def regional_average_error(out_eror):
    tropics_bndy = .25
    subtropics_north_bndy = .50
    dx = 160e3

    y = out_eror.y
    percent = 2 * y / (y.max() + dx) - 1

    subtropics = ((np.abs(percent) > tropics_bndy) &
                  (np.abs(percent) <= subtropics_north_bndy))

    tropics = np.abs(percent) <= tropics_bndy
    region = xr.where(tropics, 'Tropics',
                      xr.where(subtropics, 'Subtropics', 'Extratropics'))

    out_eror['region'] = region
    avg_ss = out_eror.groupby('region').apply(
        lambda x: (x**2).mean('y')).compute()

    return avg_ss.to_dataframe(name='SS').reset_index()


@cache
def get_data_one_field(field='QT', tlim=slice(100, 110), avg_dims='x'):
    truth = xr.open_dataset(training_data).isel(step=0)

    out_eror = {}

    for run in runs:
        pred = runs[run].data_3d[field].sel(time=tlim)
        target = truth[field]
        out_eror[run] = (
            rms(pred - target, dim=avg_dims) * truth.layer_mass).sum('z')

    out_eror = data_array_dict_to_dataset(out_eror)

    return regional_average_error(out_eror).rename(columns={'SS': field})


def get_data(keys=['U', 'V', 'SLI', 'QT']):
    data = {key: get_data_one_field(key) for key in keys}
    return reduce(pd.merge, data.values())


def plot(df):
    plotme = pd.melt(df, id_vars=["time", "region", "step", "keys"])
    sns.FacetGrid(
        plotme, row="region", col="variable", hue="keys", sharey=False)\
       .map(plt.plot, "time", "value")\
       .add_legend()

if __name__ == '__main__':
    df = get_data()
    plot(df)
