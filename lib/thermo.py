import glob
import os
from pathlib import Path

import numpy as np
import xarray as xr

import xnoah.xcalc
from xnoah.xarray import coarsen
from xnoah.xarray import map_overlap
import dask.array as da
from dask.array.core import atop
from dask import delayed

from .util import wrap_xarray_calculation, xopen, xopena

grav = 9.81
cp = 1004
Lc = 2.5104e6
rho0 = 1.19


def q2(d):
    out = -d.f_qt * Lc/cp / 1000.0
    out.attrs['units'] = 'K/d'
    return out.to_dataset(name='Q2')

def q1(d):
    out = d.f_sl
    out.attrs['units'] = 'K/d'
    return out.to_dataset(name='Q1')


@wrap_xarray_calculation
def liquid_water_temperature(t, qn, qp):
    """This is an approximate calculation neglecting ice and snow
    """

    sl = t + grav/cp * t.z - Lc/cp * (qp + qn)/1000.0
    sl.attrs['units'] = 'K'
    return sl

@wrap_xarray_calculation
def total_water(qv, qn):
    qt =  qv + qn
    qt.attrs['units'] = 'g/kg'
    return qt


def get_dz(z):
    zext = np.hstack((0,  z,  2.0*z[-1] - 1.0*z[-2]))
    zw = .5 * (zext[1:] + zext[:-1])
    dz = zw[1:] - zw[:-1]

    return xr.DataArray(dz, z.coords)

def layer_mass(rho):
    dz = get_dz(rho.z)
    return rho*dz


# Rules for Linking data
def data_files():
    root_path = Path(data_root)
    for file in root_path.glob('**/*.nc'):
        relfile = file.relative_to(root_path)
        newfile = Path(os.getcwd()).joinpath(relfile)
        yield str(file), str(newfile)

def link_files(files):
    for src, dest in files:
        path = Path(dest)
        path.parent.mkdir(exist_ok=True)
        os.system(f"ln -s {src} {path}")

