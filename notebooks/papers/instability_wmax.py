import matplotlib.pyplot as plt
import xarray as xr

from src.data import training_data, runs

plt.style.use("tableau-colorblind10")

ds = xr.open_dataset(training_data).isel(step=0)

ds['WMAX'] = ds.W.max(['x', 'y', 'z'])

time = runs['micro'].stat.time

for key, run in runs.items():
    run.stat.WMAX.plot(label=key)
ds.WMAX.interp(time=time).plot(label='NG-Aqua')
plt.legend()
