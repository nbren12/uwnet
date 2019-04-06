import xarray as xr
import numpy as np
from scipy.stats import entropy
import seaborn as sns
from uwnet.thermo import compute_apparent_source
from uwnet.stochastic_parameterization.utils import get_dataset
import matplotlib.pyplot as plt
import torch
import pandas as pd

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-m', '--model')
parser.add_argument('-n', '--ngaqua', action='store_true')

args = parser.parse_args()

pw_name = 'pw'
netprec_name = 'net_precip'


def get_data(model=torch.load(args.model)):
    # get data
    ds = get_dataset(set_eta=False)
    model.eta = model.eta[(ds.y > 4.5e6) & (ds.y < 5.5e6)]
    ds = ds.sel(
        time=slice(100, 115), y=slice(4.5e6, 5.5e6))
    # ds = ds.sel(time=slice(100, 115))
    usrf = np.sqrt(ds.U.isel(z=0)**2 + ds.V.isel(z=0)**2)
    ds['surface_wind_speed'] = usrf

    def integrate_moist(src):
        return (src * ds.layer_mass).sum('z') / 1000

    # semiprognostic
    predicted_srcs = model.predict(ds)
    ds = xr.Dataset({
        netprec_name: -integrate_moist(predicted_srcs['QT']),
        pw_name: integrate_moist(ds.QT)
    })

    return ds.to_dataframe()


def get_ng():
    ds = get_dataset(set_eta=False).sel(
        time=slice(100, 115), y=slice(4.5e6, 5.5e6))
    # ds = get_dataset(set_eta=False).sel(time=slice(100, 115))

    def integrate_moist(src):
        return (src * ds.layer_mass).sum('z') / 1000

    q2 = compute_apparent_source(ds.QT, 86400 * ds.FQT)

    ng = xr.Dataset({
        netprec_name: -integrate_moist(q2),
        pw_name: integrate_moist(ds.QT)
    })

    return ng.to_dataframe()


def hexbin(ax, df, quantiles=[.01, .1, .5, .9, .99],
           y_quantiles=[.01, .1, .5, .9, .99],
           xlim=[32, 55], ylim=[-20, 35]):
    #     levels = np.linspace(0, .04, 21)
    df = df.sample(10000)
    im = sns.kdeplot(df.pw, df.net_precip, ax=ax,
                     shade=True, shade_lowest=False)

    v = df.pw.mean()
    ax.axvline(v, ls='-', lw=1.5, c='k')
    ax.text(v - .5, 20, f'{v:.1f} mm', horizontalalignment='right', fontsize=9)

    v = df.net_precip.mean()
    ax.axhline(v, ls='-', lw=1.5, c='k')
    ax.text(
        34, v + .5, f'{v:.1f} mm/d', verticalalignment='bottom', fontsize=9)

    ax.set_xlabel('PW (mm)')
    ax.set_ylabel('Net Precip. (mm/d)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im


def calc_entropy(df):
    bins_precip = np.r_[-50:100:5]
    bins_pw = np.r_[0:55:2.5]
    H = np.histogram2d(df.pw, df.net_precip, [bins_pw, bins_precip])[0].ravel()

    df = get_ng()
    Hng = np.histogram2d(df.pw, df.net_precip, [
                         bins_pw, bins_precip])[0].ravel()

    H, Hng = H[H > 0], Hng[H > 0]
    print("Relative Entropy (compared to NG-Aqua):", entropy(Hng, H))

    print("Entropy:", entropy(H.ravel()))


def stats(df):
    bins = np.r_[30:55:2.5]
    stats_table = df.groupby(pd.cut(df.pw, bins)).net_precip.describe()
    print(stats_table)

    calc_entropy(df)

    print("Standard deviation of net precip", df.net_precip.std())
    print("Mean of net precip", df.net_precip.mean())


def plot(df, title=''):

    fig, a = plt.subplots(1, 1)
    hexbin(a, df)
    plt.title(title)


def main():

    if args.ngaqua:
        df = get_ng()
    else:
        df = get_data()
    plot(df)
    stats(df)
    plt.show()


def plot_df(df, title=''):
    plot(df)
    plot(df, title)
    plt.show()


if __name__ == '__main__':
    main()
