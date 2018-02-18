import numpy as np
import glob, re
import xarray as xr


def decode_date(units):
    match = re.search(r"(\w+) since (\d\d+)-(\d\d)-(\d\d) (\d\d:\d\d:\d\d)",
                      units)

    unit = match.group(1)
    y = match.group(2)

    if len(y) == 2:
        y = '19' + y
    elif y[:2] == '00':
        y = '19' + y[2:]
    m = match.group(3)
    d = match.group(4)
    time = match.group(5)

    return f"{unit} since {y}-{m}-{d} {time}"


def load_cam(files):
    if isinstance(files, str):
        files = glob.glob(files)
    # load data

    ds = xr.auto_combine([xr.open_dataset(f, decode_times=False)
                          for f in files[:-1]], concat_dim='time')\
           .sortby('time')
    ds.time.attrs['units'] = decode_date(ds.time.units)
    ds = xr.decode_cf(ds)
    return ds
