#!/usr/bin/env python
"""Process single time step of NGAqua Data for neural network"""
import os
import shutil
import tempfile

import click
from toolz import assoc_in

import xarray as xr
from sam.case import InitialConditionCase, default_parameters, get_ngqaua_ic

ngaqua_root = "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"
sam_src = "/Users/noah/workspace/models/SAMUWgh"


def get_parameters(n, dt=30.0):
    prm = default_parameters
    for key, val in [('usepython', False), ('dopython', False), ('nstop', n),
                     ('nsave3d', n), ('nstat', n), ('nstatfrq', 1), ('dt', dt),
                     ('nsave3dstart', 0)]:

        prm = assoc_in(prm, ['parameters', key], val)
    return prm


def run_sam_nsteps(ic, n=10, dt=30.0, sam_src=sam_src):
    path = tempfile.mkdtemp(dir=".")
    case = InitialConditionCase(
        ic=ic, sam_src=sam_src, prm=get_parameters(n, dt), path=path)
    case.run_docker()
    case.convert_files_to_netcdf()

    return path


@click.command()
@click.option('-n', '--ngaqua-root', type=click.Path(), default=ngaqua_root)
@click.argument('t', type=int)
def main(ngaqua_root, t):
    dt = 30.0
    n = 10
    ic = get_ngqaua_ic(ngaqua_root, t)

    path = run_sam_nsteps(ic, n=n, dt=dt, sam_src=sam_src)
    files = os.path.join(path, 'OUT_3D', '*.nc')

    # open files
    ds = xr.open_mfdataset(files)

    # assign coords from IC
    time = ic.time
    ic = ic.drop('time')
    ds = ds.assign_coords(**ic.coords).drop('time')

    training_data = xr.Dataset({
        'U': ds.U[0],
        'V': ds.V[0],
        'W': ds.W[0],
        'QN': ic.QN,
        'QP': ic.QP,
        'QT': ds.QT[0],
        'SLI': ds.SLI[0],
        'FV': (ds.V[1] - ds.V[0]) / n / dt,
        'FU': (ds.U[1] - ds.U[0]) / n / dt,
        'FSLI': (ds.SLI[1] - ds.SLI[0]) / n / dt,
        'FQT': (ds.QT[1] - ds.QT[0]) / n / dt,
        'time': time,
    })

    # save training data to netcdf
    out_path = "%06d.nc" % t
    training_data.expand_dims('time')\
                 .load()\
                 .to_netcdf(out_path)

    # clean up SAM run
    shutil.rmtree(path)


if __name__ == '__main__':
    main()
