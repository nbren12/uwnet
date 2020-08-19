# coding: utf-8
import xarray as xr
import matplotlib.pyplot as plt

plt.style.use('presentation')

xr.open_dataset("data/binned.nc")
bins = xr.open_dataset("data/binned.nc")
varyq = bins.mean('lts_bins')
varyq.net_precipitation_nn.plot()
varyq.net_precipitation_src.plot()


plt.xlabel('Mid-tropospheric Moisture (mm)')
plt.title("Precipitation - Evaporation (mm/day)")
plt.ylabel('')
plt.legend(['Neural network', 'Truth'])

plt.tight_layout()-