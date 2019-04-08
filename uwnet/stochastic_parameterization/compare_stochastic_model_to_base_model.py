import xarray as xr
import matplotlib.pyplot as plt
from uwnet.stochastic_parameterization.utils import get_dataset
from uwnet.thermo import lhf_to_evap


dir_ = '/Users/stewart/Desktop/'
# ds_s = xr.open_dataset(dir_ + 'stochastic_model_gcm_output.nc')
ds_s = xr.open_dataset(dir_ + 'no_hyper_diffuse.nc')
ds_b = xr.open_dataset(dir_ + 'base_model_gcm_output.nc')
ds_true = get_dataset(
    ds_location="/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/training.nc",  # noqa
    set_eta=False,
    t_start=0,
    t_stop=len(ds_s.time))
ds_true['NPNN'] = ds_true.Prec - lhf_to_evap(ds_true.LHF)


def plot_pw_over_time():
    plt.plot(ds_s.PW.mean(['x', 'y']), label='Stochastic Model')
    plt.plot(ds_b.PW.mean(['x', 'y']), label='Base Model')
    plt.plot(ds_true.PW.mean(['x', 'y']), label='True Data')
    plt.ylabel('Mean Precipital Water (mm)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Global Mean Precipital Water over Time')
    plt.show()


def plot_npnn_over_time_equator():
    ds_s.isel(y=32).NPNN.mean('x').plot(label='Stochastic Model')
    ds_b.isel(y=32).NPNN.mean('x').plot(label='Base Model')
    ds_true.isel(y=32).NPNN.mean('x').plot(label='True Data')
    plt.ylabel('Equator Mean Precipital Water (mm)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Equator Mean Net Precip over Time')
    plt.show()


def plot_zonal_mean_net_precip_over_time():
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    ds_s.NPNN.mean('x').plot(x='time', ax=axs[0])
    axs[0].set_title('Stochastic Model')
    ds_b.NPNN.mean('x').plot(x='time', ax=axs[1], label='Base Model')
    axs[1].set_title('Base Model')
    ds_true.NPNN.mean('x').plot(x='time', ax=axs[2], label='True Data')
    axs[2].set_title('True Data')
    fig.suptitle('Zonal Mean Net Precip Over Time')
    fig.subplots_adjust(hspace=.5)
    plt.show()


def plot_pw_tropics_zonal_variance_over_time():
    ds_s.isel(y=range(28, 36)).NPNN.mean('y').var('x').plot(
        label='Stochastic Model')
    ds_b.isel(y=range(28, 36)).NPNN.mean('y').var('x').plot(label='Base Model')
    ds_true.isel(y=range(28, 36)).NPNN.mean('y').var('x').plot(
        label='True Data')
    plt.ylabel('Equator Zonal Variance in Net Precipitation (mm)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Equator Zonal Variance Net Precipitation over Time')
    plt.show()


if __name__ == '__main__':
    plot_pw_over_time()
    plot_npnn_over_time_equator()
    plot_zonal_mean_net_precip_over_time()
    plot_pw_tropics_zonal_variance_over_time()
