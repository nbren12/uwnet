import numpy as np
import glob
import xarray as xr


def load_cam(files):
    if isinstance(files, str):
        files = glob.glob(files)
    # load data
    ds = xr.auto_combine([xr.open_dataset(f, decode_times=False)
                          for f in files[:-1]], concat_dim='time')\
           .sortby('time')
    ds.time.attrs['units'] = 'days since 1999-04-11 15:00:00'
    ds = xr.decode_cf(ds)
    time = ds.time.values - np.datetime64('1999-01-01')
    time = time.astype(dtype='timedelta64[s]').astype(int)/86400
    ds.coords['time'] = time
    return ds
