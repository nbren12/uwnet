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
    return xr.decode_cf(ds)
