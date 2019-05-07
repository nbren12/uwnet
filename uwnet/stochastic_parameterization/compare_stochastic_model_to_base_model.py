import xarray as xr
import matplotlib.pyplot as plt
from uwnet.stochastic_parameterization.utils import get_dataset
from uwnet.thermo import lhf_to_evap


plt.style.use('tableau-colorblind10')


def plot_total_pw_over_time(
        ds_s, ds_b, ds_true, ds_no_parameterization=None):
    plt.plot(ds_s.isel(y=range(28, 36)).PW.mean(
        ['x', 'y']), label='Stochastic Model')
    plt.plot(ds_b.isel(y=range(28, 36)).PW.mean(
        ['x', 'y']), label='Base Model')
    plt.plot(ds_true.isel(y=range(28, 36)).PW.mean(
        ['x', 'y']), label='True Data')
    if ds_no_parameterization:
        plt.plot(
            ds_no_parameterization.isel(y=range(28, 36)).PW.mean(['x', 'y']),
            label='No Parameterization')
    plt.ylabel('Mean PW (mm) in Tropics')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Tropics Mean Precipital Water over Time')
    plt.show()


def plot_npnn_over_time(ds_s, ds_b, ds_true, ds_no_parameterization=None):
    ds_s.isel(y=range(28, 36)).NPNN.mean(['x', 'y']).plot(
        label='Stochastic Model')
    ds_b.isel(y=range(28, 36)).NPNN.mean(['x', 'y']).plot(label='Base Model')
    ds_true.isel(y=range(28, 36)).NPNN.mean(['x', 'y']).plot(label='True Data')
    plt.ylabel('Mean NPNN in Tropics (mm/day)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Mean NPNN in Tropics over Time')
    plt.show()


def plot_zonal_mean_npnn_over_time(ds_s, ds_b, ds_true):
    n_figs = 3
    fig, axs = plt.subplots(
        n_figs, 1, sharex=True, sharey=True)
    ds_s.NPNN.mean('x').plot(x='time', ax=axs[0])
    axs[0].set_title('Stochastic Model')
    ds_b.NPNN.mean('x').plot(x='time', ax=axs[1], label='Base Model')
    axs[1].set_title('Base Model')
    ds_true.NPNN.mean('x').plot(x='time', ax=axs[2], label='True Data')
    axs[2].set_title('True Data')
    fig.suptitle('Zonal Mean Net Precip Over Time')
    fig.subplots_adjust(hspace=.5)
    plt.show()


def plot_pw_tropics_zonal_variance_over_time(
        ds_s, ds_b, ds_true, ds_no_parameterization=None):
    ds_s.isel(y=range(28, 36)).PW.mean('y').var('x').plot(
        label='Stochastic Model')
    ds_b.isel(y=range(28, 36)).PW.mean('y').var('x').plot(label='Base Model')
    ds_true.isel(y=range(28, 36)).PW.mean('y').var('x').plot(
        label='True Data')
    if ds_no_parameterization:
        ds_no_parameterization.isel(
            y=range(28, 36)).PW.mean('y').var('x').plot(label='True Data')
    plt.ylabel('Equator Zonal Variance in PW (mm)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Equator Zonal Variance PW over Time')
    plt.show()


def plot_u_rmse_over_time(
        ds_s, ds_b, ds_true, ds_no_param, ds_no_parameterization=None):
    stochastic_model_error = ((
            ds_s.USFC.values - ds_true.isel(z=0).U.values) ** 2).mean(
            axis=1).mean(axis=1)
    base_model_error = ((
            ds_b.USFC.values - ds_true.isel(z=0).U.values) ** 2).mean(
            axis=1).mean(axis=1)
    no_param_model_error = ((
            ds_no_param.USFC.values - ds_true.isel(z=0).U.values) ** 2).mean(
            axis=1).mean(axis=1)
    plt.plot(base_model_error, label='Base Model')
    plt.plot(stochastic_model_error, label='Stochastic Model')
    plt.plot(no_param_model_error, label='No Parameterization')
    plt.legend(loc='best')
    plt.title('USFC Global RMSE Over Time')
    plt.show()


def plot_v_rmse_over_time(
        ds_s, ds_b, ds_true, ds_no_param, ds_no_parameterization=None):
    stochastic_model_error = ((
            ds_s.VSFC.values - ds_true.isel(z=0).U.values) ** 2).mean(
            axis=1).mean(axis=1)
    base_model_error = ((
            ds_b.VSFC.values - ds_true.isel(z=0).U.values) ** 2).mean(
            axis=1).mean(axis=1)
    no_param_model_error = ((
            ds_no_param.VSFC.values - ds_true.isel(z=0).U.values) ** 2).mean(
            axis=1).mean(axis=1)
    plt.plot(base_model_error, label='Base Model')
    plt.plot(stochastic_model_error, label='Stochastic Model')
    plt.plot(no_param_model_error, label='No Parameterization')
    plt.legend(loc='best')
    plt.title('VSFC Global RMSE Over Time')
    plt.show()


def plot_rmse_over_time(ds_s, ds_b, ds_true, no_param, var):
    stochastic_model_error = ((
            ds_s[var].values - ds_true[var].values) ** 2).mean(
            axis=1).mean(axis=1)
    base_model_error = ((
            ds_b[var].values - ds_true[var].values) ** 2).mean(
            axis=1).mean(axis=1)
    no_param_model_error = ((
            no_param[var].values - ds_true[var].values) ** 2).mean(
            axis=1).mean(axis=1)
    plt.plot(base_model_error, label='Base Model')
    plt.plot(stochastic_model_error, label='Stochastic Model')
    plt.plot(no_param_model_error, label='No Parameterization')
    plt.title(f'{var} Glabal RMSE Over Time')
    plt.ylabel(f'{var} Global RMSE')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    dir_ = '/Users/stewart/Desktop/'
    ds_no_param = xr.open_dataset(dir_ + 'no_parameterization.nc')
    ds_s = xr.open_dataset(dir_ + 'stochastic_model_gcm_output_long_run.nc')
    # ds_s = xr.open_dataset(dir_ + 'stochastic_model_gcm_output.nc')
    # ds_s = xr.open_dataset(dir_ + 'no_hyper_diffuse.nc')
    ds_b = xr.open_dataset(dir_ + 'base_model_gcm_output_long_run.nc')
    # ds_b = xr.open_dataset(dir_ + 'base_model_gcm_output.nc')
    # ds_b = xr.open_dataset(dir_ + 'no_hyper_diffuse_base_model.nc')
    ds_true = get_dataset(
        ds_location="/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/training.nc",  # noqa
        set_eta=False,
        t_start=0,
        t_stop=len(ds_s.time))
    ds_true['NPNN'] = ds_true.Prec - lhf_to_evap(ds_true.LHF)

    # plot_u_rmse_over_time(ds_s, ds_b, ds_true, ds_no_param)
    # plot_v_rmse_over_time(ds_s, ds_b, ds_true, ds_no_param)
    # plot_rmse_over_time(ds_s, ds_b, ds_true, ds_no_param, 'PW')
    # plot_rmse_over_time(ds_s, ds_b, ds_true, ds_no_param, 'NPNN')
    plot_total_pw_over_time(ds_s, ds_b, ds_true)
    plot_npnn_over_time(ds_s, ds_b, ds_true)
    plot_zonal_mean_npnn_over_time(ds_s, ds_b, ds_true)
    plot_pw_tropics_zonal_variance_over_time(ds_s, ds_b, ds_true)
