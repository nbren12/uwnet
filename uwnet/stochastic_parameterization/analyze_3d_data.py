from matplotlib import pyplot as plt
import xarray as xr
from uwnet.stochastic_parameterization.utils import get_dataset
from uwnet.thermo import integrate_q2
from uwnet.stochastic_parameterization.graph_utils import loghist

ds_run = xr.open_dataset('/Users/stewart/Desktop/stochastic_out_3d.nc')
ds_true = get_dataset(t_start=0, t_stop=80)

ds_run = ds_run.isel(y=range(28, 36))
ds_true = ds_true.isel(y=range(28, 36))

for time_idx in range(1, min(len(ds_run.time), len(ds_true.time))):
    fig, ax = plt.subplots()
    loghist(
        -integrate_q2(
            ds_true.isel(time=time_idx).QT, ds_true.layer_mass).values.ravel(),
        ax=ax,
        upper_percentile=99.99,
        lower_percentile=0.01,
        label='True Data',
        gaussian_comparison=False
    )
    loghist(
        -integrate_q2(
            ds_run.isel(time=time_idx).QT, ds_true.layer_mass).values.ravel(),
        ax=ax,
        upper_percentile=99.99,
        lower_percentile=0.01,
        label='Simulation with Stochastic Model',
        gaussian_comparison=False
    )
    plt.legend()
    plt.title(f'QT Comparison for Time = {ds_run.time[time_idx].values}')
    plt.show()
