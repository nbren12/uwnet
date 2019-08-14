import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from uwnet.stochastic_parameterization.choose_bins import \
    optimize_binning_quantiles
import xarray as xr
from uwnet.stochastic_parameterization.utils import get_dataset
from uwnet.thermo import (
    compute_apparent_source,
    liquid_water_density,
    cp,
    sec_in_day,
)

style = 'viridis'


dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
desktop_ = '/Users/stewart/Desktop/'
ds_location = dir_ + 'training.nc'
ds = xr.open_dataset(ds_location).isel(time=range(80))


def net_precip_comparisons(model_s, model_b):
    ds_s = xr.open_dataset(desktop_ + 'stochastic_model_gcm_output_good.nc')
    time = ds_s.isel(time=16).time
    ds_b = xr.open_dataset(desktop_ + 'base_model_gcm_output.nc')
    ds_true = get_dataset(
        ds_location=dir_ + "training.nc",
        base_model_location=dir_ + 'full_model/1.pkl',
        # set_eta=True,
        t_start=0,
        t_stop=len(ds_b.time))
    true = -(compute_apparent_source(
        ds_true.QT, ds_true.FQT * 86400).sel(time=time).dot(
        ds_true.layer_mass) / liquid_water_density
    )
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    true.plot(
        ax=ax[0][0],
        add_colorbar=False,
        vmin=true.min() - 1,
        vmax=true.max() + 1)
    ax[0][0].title.set_text('NG-Aqua')
    pred = ds_b.NPNN.sel(time=time)
    pred.attrs['units'] = 'mm/day'
    pred.attrs['long_name'] = 'Net Precip.'
    pred = pred.rename('Net Precip.')
    pred.plot(
        ax=ax[0][1],
        add_colorbar=False,
        vmin=true.min() - 1,
        vmax=true.max() + 1)
    ax[0][1].title.set_text('Non-Stochastic Simulation')
    pred = ds_s.NPNN.sel(time=time)
    pred.attrs['units'] = 'mm/day'
    pred.attrs['long_name'] = 'Net Precip.'
    pred = pred.rename('Net Precip.')
    pred.plot(
        ax=ax[0][2],
        # add_colorbar=False,
        vmin=true.min() - 1,
        vmax=true.max() + 1)
    ax[0][2].title.set_text('Stochastic Simulation')
    fig.suptitle('Net Precip. Comparisons: Simulation vs Offline at Day 2')

    true.plot(
        ax=ax[1][0],
        add_colorbar=False,
        vmin=true.min() - 1,
        vmax=true.max() + 1)
    ax[1][0].title.set_text('NG-Aqua')

    pred_b = -(model_b.predict(ds.sel(time=[time]))['QT'].dot(
        ds.layer_mass).isel(time=0) / liquid_water_density)
    pred_b.attrs['units'] = 'mm/day'
    pred_b.attrs['long_name'] = 'Net Precip.'
    pred_b = pred_b.rename('Net Precip.')
    pred_b.plot(
        ax=ax[1][1],
        add_colorbar=False,
        vmin=true.min() - 1,
        vmax=true.max() + 1,
        cmap=style)
    ax[1][1].title.set_text('Non-Stochastic Model Offline')
    # eta=ds_true.sel(time=time).eta.values
    model_s.eta = ds_true.sel(time=time - .125).eta.values
    pred_s = -(model_s.predict(
        ds.sel(time=[time]))['QT'].dot(
        ds.layer_mass).isel(time=0) / liquid_water_density)
    pred_s.attrs['units'] = 'mm/day'
    pred_s.attrs['long_name'] = 'Net Precip.'
    pred_s = pred_s.rename('Net Precip.')
    pred_s.plot(
        ax=ax[1][2],
        vmin=true.min() - 1,
        vmax=true.max() + 1,
        # add_colorbar=False,
        cmap=style)
    ax[1][2].title.set_text('Stochastic Model Offline')
    plt.show()


def plot_net_precip(model_s, model_b, ds):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    true = -(compute_apparent_source(
        ds.QT, ds.FQT * 86400).isel(time=0).dot(ds.layer_mass) /
        liquid_water_density
    )
    true.plot(
        ax=ax[0],
        add_colorbar=False,
        vmin=true.min() - 1,
        cmap=style,
        vmax=true.max() + 1)
    ax[0].title.set_text('NG-Aqua Net Precip')
    pred_b = -(model_b.predict(ds.isel(time=[0]))['QT'].dot(
        ds.layer_mass).isel(time=0) / liquid_water_density)
    pred_b.attrs['units'] = 'mm/day'
    pred_b.attrs['long_name'] = 'Net Precip.'
    pred_b = pred_b.rename('Net Precip.')
    pred_b.plot(
        ax=ax[1],
        add_colorbar=False,
        vmin=true.min() - 1,
        vmax=true.max() + 1,
        cmap=style)

    # pred_s = -(model_s.predict(ds.isel(time=[16]))['QT'].dot(
    #     ds.layer_mass).isel(time=0) / liquid_water_density)
    # pred_s.attrs['units'] = 'mm/day'
    # pred_s.attrs['long_name'] = 'Net Precip.'
    # pred_s = pred_s.rename('Net Precip.')
    # pred_s.plot(
    #     ax=ax[2],
    #     vmin=true.min() - 1,
    #     vmax=true.max() + 1,
    #     # add_colorbar=False,
    #     cmap=style)
    # ax[2].title.set_text('Stochastic Model Net Precip')
    fig.suptitle('Net Precip. Comparison')
    # fig.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()


def plot_net_heating(model, ds):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    true = (compute_apparent_source(
        ds.SLI, ds.FSLI * 86400).isel(time=0).dot(ds.layer_mass)
        * (cp / sec_in_day)
    )
    true.plot(
        ax=ax[0],
        add_colorbar=False,
        vmin=true.min() - 1,
        cmap=style,
        vmax=true.max() + 1)
    ax[0].title.set_text('NG-Aqua Net Heating')
    pred = (model.predict(ds.isel(time=[0]))['SLI'].dot(
        ds.layer_mass).isel(time=0) * (cp / sec_in_day))
    pred.attrs['units'] = 'W/m2'
    pred.attrs['long_name'] = 'Net Heating'
    pred = pred.rename('Net Heating')
    pred.plot(
        ax=ax[1],
        vmin=true.min() - 1,
        vmax=true.max() + 1,
        # add_colorbar=False,
        cmap=style)
    ax[1].title.set_text('NN-Predicted Net Heating')
    fig.suptitle('Net Heating Comparison')
    plt.show()


def plot_2d(var):
    ds[var].isel(time=0).plot()
    long_name = ds[var].attrs['long_name']
    plt.title(f'NG-Aqua {long_name} at Time = 0')
    plt.show()


def plot_2d_over_time():
    for var in ['SST', 'SOLIN']:
        long_name = ds[var].attrs['long_name']
        ds[var].isel(y=32).mean('x').plot()
        plt.title(f'NG-Aqua {long_name} at Equator vs Time')
        plt.show()


def plot_2d_vars():
    for var in [
        'Prec',
        # 'LHF',
        # 'SHF',
        # 'SOLIN',
        # 'SST',
    ]:
        plot_2d(var)


def plot_solin():
    long_name = ds['SOLIN'].attrs['long_name']
    fig, ax = plt.subplots(1, 2)
    ds['SOLIN'].isel(time=0).plot(ax=ax[0])
    ds['SOLIN'].isel(y=32).mean('x').plot(ax=ax[1])
    ax[0].title.set_text(f'NG-Aqua {long_name} at Time = 0')
    ax[1].title.set_text(f'NG-Aqua {long_name} at Equator vs Time')
    plt.show()


def plot_3d_vars():
    for var in [
        # 'FQT',
        'FSLI',
        # 'FU'
        # 'QT',
        # 'SLI',
        # 'U',
        # 'V',
        # 'W'
    ]:
        fig, ax = plt.subplots(1, 2)
        ds[var].isel(time=0, z=10, y=range(1, 63)).plot(ax=ax[0])
        ax[0].set_title(f'Layer of NG-Aqua {var} (Time=0, z=1300m)', y=1.08)
        ds[var].isel(time=0).isel(y=32).plot(ax=ax[1])
        ax[1].set_title(
            f'NG-Aqua Vertical Slice of {var} at Equator (Time=0)', y=1.08)
        plt.subplots_adjust(wspace=.35)
        plt.show()


def plot_optimal_bins():
    n_bins = 7
    binning_quantiles = optimize_binning_quantiles(n_bins, verbose=False)
    print(f'Binning Quantiles: {np.round(binning_quantiles, 4)}')
    ax = plt.axes()
    p = gaussian_kde(ds)
    lower_percentile = 0.001
    upper_percentile = 99.999
    p1 = np.percentile(ds, lower_percentile)
    p2 = np.percentile(ds, upper_percentile)
    xx = np.linspace(p1, p2, 100)
    y = np.log(p(xx))
    ax.plot(xx, y)
    ax.set_xlim([p1, p2])
    bin_values = [np.quantile(ds, quantile) for quantile in binning_quantiles]
    for bin_value in bin_values:
        plt.axvline(x=bin_value, col='r')
    plt.title(
        f'Optimal {n_bins} bins superimposed on column integrated QT residuals'
    )
    plt.ylabel('Log Density')
    plt.xlabel('Column Integrated QT Residual')
    plt.show()


if __name__ == '__main__':
    base_model = torch.load(dir_ + 'base_model.pkl')
    stochastic_model = torch.load(dir_ + 'stochastic_model.pkl')
    # plot_net_heating(base_model, ds)
    # plot_2d_vars()
    # plot_2d_over_time()
    # plot_net_precip(stochastic_model, base_model, ds)
    plot_3d_vars()
    # net_precip_comparisons(stochastic_model, base_model)
    # plot_solin()
    plot_optimal_bins()
