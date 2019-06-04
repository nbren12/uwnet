import torch
from matplotlib import pyplot as plt
import xarray as xr
from uwnet.stochastic_parameterization.utils import (
    get_dataset,
)
from uwnet.thermo import (
    compute_apparent_source,
    liquid_water_density,
    cp,
    sec_in_day,
)

style = 'viridis'


dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
ds_location = dir_ + 'training.nc'
ds = xr.open_dataset(ds_location).isel(time=range(10))


def plot_net_precip(model, ds):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    true = -(compute_apparent_source(
        ds.QT, ds.FQT * 86400).isel(time=0).dot(ds.layer_mass)
        / liquid_water_density
     )
    true.plot(
        ax=ax[0],
        add_colorbar=False,
        vmin=-100,
        cmap=style,
        vmax=100)
    ax[0].title.set_text('NG-Aqua Net Precip')
    pred = -(model.predict(ds.isel(time=[0]))['QT'].dot(
        ds.layer_mass).isel(time=0) / liquid_water_density)
    pred.attrs['units'] = 'mm/day'
    pred.attrs['long_name'] = 'Net Precip.'
    pred = pred.rename('Net Precip.')
    pred.plot(
        ax=ax[1],
        vmin=-100,
        vmax=100,
        cmap=style)
    ax[1].title.set_text('NN-Predicted Net Precip')
    fig.suptitle('Net Precip Comparison')
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
        vmin=-600,
        cmap=style,
        vmax=3200)
    ax[0].title.set_text('NG-Aqua Net Heating')
    pred = (model.predict(ds.isel(time=[0]))['SLI'].dot(
        ds.layer_mass).isel(time=0) * (cp / sec_in_day))
    pred.attrs['units'] = 'W/m2'
    pred.attrs['long_name'] = 'Net Heating'
    pred = pred.rename('Net Heating')
    pred.plot(
        ax=ax[1],
        vmin=-600,
        vmax=3200,
        cmap=style)
    ax[1].title.set_text('NN-Predicted Net Heating')
    fig.suptitle('Net Heating Comparison')
    plt.show()


if __name__ == '__main__':
    model = torch.load(dir_ + 'base_model.pkl')
    plot_net_precip(model, ds)
    # plot_net_heating(model, ds)
