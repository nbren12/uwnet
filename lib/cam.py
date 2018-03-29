import glob
import re

import numpy as np

import xarray as xr
from lib.interp import interp
from xnoah import swap_coord


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

    return unit, np.datetime64(f"{y}-{m}-{d} {time}")


def load_cam(files):
    if isinstance(files, str):
        files = glob.glob(files)
    # load data

    ds = xr.auto_combine([xr.open_dataset(f, decode_times=False)
                          for f in files[:-1]], concat_dim='time')\
           .sortby('time')

    unit, bdate = decode_date(ds.time.units)
    if unit == 'days':
        # xarray needs the dtype to be timedelat64[ns]
        # when saving the netcdf file
        time = np.timedelta64(int(86400*1e9), 'ns') * ds.time.values + bdate
    ds = ds.assign_coords(time=time)
    return ds


def convert_dates_to_days(x, bdate, dim='time'):
    if isinstance(bdate, str):
        bdate = np.datetime64(bdate, dtype='datetime64[s]')
    time = x[dim].values - bdate
    time = time.astype('timedelta64[s]').astype(float) / 86400
    return x.assign_coords(**{dim: time})


def hybrid_to_pres(hya, hyb, p0, ps):
    return hya * p0 + hyb * ps


def to_sam_z(cam, p0, dim='lev'):
    """Interpolate cam onto heights from sam"""

    # load and interpolate CAM
    # cam = xr.open_dataset(root / "output/scam.nc")
    p = hybrid_to_pres(cam.hyam, cam.hybm, cam.P0, cam.PS)/100

    def _interp(x):
        if dim in x.dims:
            return interp(p0, p, x, old_dim=dim, log=True)
        else:
            return x
    cam = cam.apply(_interp).assign(z=p0.z)
    cam = swap_coord(cam, {dim: 'z'})
    cam = cam.assign(p=p0)
    return cam
