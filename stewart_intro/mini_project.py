"""
use FQT, which is the output of the Fortran model at time t, and iteratively
step Fortran model forward for N time steps and compare the results to true
model at time t + N
"""
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

file_location = '~/projects/2018-09-18-NG_5120x2560x34_4km_10s_QOBS_EQX-SAM_Processed.nc'  # noqa

data = xr.open_dataset(file_location)

dt = 60 * 60 * 3  # 3 hours in seconds

n_steps = 2

fqt_errors = np.zeros(
    (len(data.time) - 1, len(data.z), len(data.y), len(data.x))
)

baseline_errors = fqt_errors.copy()

# for idx, time in enumerate(data.time.values[n_steps:]):

estimates = []
qt_est = data.isel(time=0).QT
for i in range(1, 101):
    qt_est = qt_est + (dt * data.isel(time=i).FQT)
    estimates.append(qt_est.values)
est_array = np.stack(estimates)

to_plot = est_array[:, :, 32, 33]

plt.pcolormesh(to_plot.T)
plt.contourf(to_plot.T)
plt.colorbar()
plt.show()

#     t_start = data.time.values[idx]
#     t_start_plus_1 = time
#     qt_est = data.sel(time=t_start).QT + (dt * data.sel(time=t_start).FQT)
#     fqt_errors[idx, :, :, :] = data.sel(time=t_start_plus_1).QT - qt_est
#     baseline_errors[idx, :, :, :] = data.sel(
#         time=t_start_plus_1).QT - data.sel(time=t_start).QT

# print('Baseline error: {}'.format(np.linalg.norm(baseline_errors)))
# print('FQT error: {}'.format(np.linalg.norm(fqt_errors)))
