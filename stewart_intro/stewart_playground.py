import matplotlib.pyplot as plt
import xarray as xr

data = xr.open_dataset(
    '../2018-09-18-NG_5120x2560x34_4km_10s_QOBS_EQX-SAM_Processed.nc')

# data.Prec.isel(time=0).plot()
data.QT.isel(x=40, y=32, time=slice(0, 100)).plot(y="z")

plt.show()
