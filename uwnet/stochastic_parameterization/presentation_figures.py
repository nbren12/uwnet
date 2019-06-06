import torch
from matplotlib import pyplot as plt
import xarray as xr
from uwnet.thermo import (
    compute_apparent_source,
    liquid_water_density,
    cp,
    sec_in_day,
)

style = 'viridis'


dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
ds_location = dir_ + 'training.nc'
ds = xr.open_dataset(ds_location).isel(time=range(80))


def plot_net_precip(model_s, model_b, ds):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
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

    pred_s = -(model_s.predict(ds.isel(time=[0]))['QT'].dot(
        ds.layer_mass).isel(time=0) / liquid_water_density)
    pred_s.attrs['units'] = 'mm/day'
    pred_s.attrs['long_name'] = 'Net Precip.'
    pred_s = pred_s.rename('Net Precip.')
    pred_s.plot(
        ax=ax[2],
        vmin=true.min() - 1,
        vmax=true.max() + 1,
        # add_colorbar=False,
        cmap=style)
    ax[2].title.set_text('Stochastic Model Net Precip')
    fig.suptitle('Net Precip Comparison')
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
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


def plot_3d_vars():
    for var in [
        'FQT',
        'FSLI',
        'FU'
        # 'QT',
        # 'SLI',
        # 'U',
        # 'V',
        # 'W'
    ]:
        ds[var].isel(time=0).dot(ds.layer_mass).plot()
        plt.title(f'NG-Aqua Column-Integrated {var} at Time = 0')
        plt.show()
        ds[var].isel(time=0).isel(y=32).plot()
        plt.title(f'NG-Aqua Vertical Slice of {var} at Equator (Time=0)')
        plt.show()


if __name__ == '__main__':
    base_model = torch.load(dir_ + 'base_model.pkl')
    stochastic_model = torch.load(dir_ + 'stochastic_model.pkl')
    # plot_net_heating(base_model, ds)
    # plot_2d_vars()
    # plot_2d_over_time()
    plot_net_precip(stochastic_model, base_model, ds)
    # plot_3d_vars()
