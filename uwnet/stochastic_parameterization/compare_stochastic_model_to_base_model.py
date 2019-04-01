import xarray as xr
import matplotlib.pyplot as plt


dir_ = '/Users/stewart/Desktop/'
ds_s = xr.open_dataset(dir_ + 'stochastic_model_gcm_output.nc')
ds_b = xr.open_dataset(dir_ + 'base_model_gcm_output.nc')


def plot_pw_over_time():
    plt.plot(ds_s.PW.mean(['x', 'y']), label='Stochastic Model')
    plt.plot(ds_b.PW.mean(['x', 'y']), label='Base Model')
    plt.ylabel('Mean Precipital Water (mm)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Global Mean Precipital Water over Time')
    plt.show()


def plot_npnn_over_time_equator():
    ds_s.isel(y=32).NPNN.mean('x').plot(label='Stochastic Model')
    ds_b.isel(y=32).NPNN.mean('x').plot(label='Base Model')
    plt.ylabel('Equator Mean Precipital Water (mm)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Equator Mean Net Precip over Time')
    plt.show()


def plot_zonal_mean_net_precip_over_time():
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    ds_s.NPNN.mean('x').plot(x='time', ax=axs[0])
    axs[0].set_title('Stochastic Model')
    ds_b.NPNN.mean('x').plot(x='time', ax=axs[1], label='Base Model')
    axs[1].set_title('Base Model')
    fig.suptitle('Zonal Mean Net Precip Over Time')
    plt.show()


def plot_pw_tropics_zonal_variance_over_time():
    ds_s.isel(y=range(28, 36)).NPNN.mean('y').var('x').plot(
        label='Stochastic Model')
    ds_b.isel(y=range(28, 36)).NPNN.mean('y').var('x').plot(label='Base Model')
    plt.ylabel('Equator Zonal Variancein Net Precipital Water (mm)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Equator Zonal Variance Net Precipital Water over Time')
    plt.show()
