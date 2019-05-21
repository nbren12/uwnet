"""Routines for opening the NGAqua Dataset"""
import xarray as xr
import os
from os.path import join
from uwnet import thermo


class NGAqua:
    def __init__(self, basedir):
        self.basedir = basedir

    @property
    def data_2d(self):
        path = join(self.basedir, "coarse", "2d", "all.nc")
        ds = xr.open_dataset(path, chunks={'time': 1})#.sortby('time')
        ds = ds.assign(NPNN=thermo.net_precipitation_from_prec_evap(ds),
                       NHNN=thermo.net_heating_from_data_2d(ds))
        # adjust coordinates
        ds['x'] -= ds.x[0]
        ds['y'] -= ds.y[0]
        return ds

    @property
    def stat(self):
        stat_path = join(self.basedir, 'stat.nc')
        return xr.open_dataset(stat_path)

    @property
    def times_3d(self):
        path = join(self.basedir, "coarse", "3d", "all.nc")
        ds = xr.open_dataset(path, cache=False)
        return list(ds.time)


def _rename_var(z, coords):
    rename_d = {'xc': 'x', 'yc': 'y', 'ys': 'y', 'xs': 'x'}
    rename_d = {key: val for key, val in rename_d.items() if key in z.dims}

    return z.rename(rename_d).assign_coords(**coords)


def open(basedir):
    """Get an initial condition for time t from the NGAQUA"""
    stagger_path = os.path.join(basedir, "stagger", "3d", "all.nc")
    center_path = os.path.join(basedir, "coarse", "3d", "all.nc")
    stat_path = os.path.join(basedir, "stat.nc")

    # open necessary files
    cent = xr.open_dataset(center_path, engine='netcdf4')
    time = cent.time
    stag = (xr.open_dataset(stagger_path).sel(time=time)
            .apply(lambda x: _rename_var(x, cent.coords)))
    stat = xr.open_dataset(stat_path)

    ic = xr.Dataset({
        'U': stag.U,
        'V': stag.V,
        'W': cent.W,
        'QV': cent.QV,
        'TABS': cent.TABS,
        'QN': cent.QN,
        'QP': cent.QP,
        'RHO': stat.RHO[0].drop('time'),
        'Ps': stat.Ps[0].drop('time')
    })

    return ic
