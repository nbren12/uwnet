"""Routines for opening the NGAqua Dataset"""
import xarray as xr
import os


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
