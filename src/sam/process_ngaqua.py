#!/usr/bin/env python
"""Process single time step of NGAqua Data for neural network"""
import os
from os.path import abspath
import shutil
import tempfile
from contextlib import contextmanager
import json

import click

import xarray as xr
from .case import InitialConditionCase, default_parameters, get_ngqaua_ic

NGAQUA_ROOT = "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"


@contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def get_parameters(n, dt=30.0):
    prm = default_parameters()

    prm['parameters']['nstop'] = n
    prm['parameters']['nsave3d'] = n
    prm['parameters']['nsave3dstart'] = 0
    prm['parameters']['nstat'] = n
    prm['parameters']['nstatfrq'] = 1
    prm['parameters']['dt'] = dt
    prm['parameters']['dosgsthermo'] = False
    prm['python']['dopython'] = False
    return prm


def run_sam_nsteps(ic, prm, sam_src):
    path = tempfile.mkdtemp(dir=".")
    case = InitialConditionCase(
        ic=ic,
        prm=prm,
        path=path,
        sam_src=sam_src)

    case.save()
    case.run()
    return path


@click.command()
@click.option('-n', '--ngaqua-root', type=click.Path(), default=NGAQUA_ROOT)
@click.argument('t', type=int)
@click.argument('out_path', type=click.Path())
@click.option('--sam', type=click.Path(), default='/opt/sam')
@click.option('-p', '--parameters', type=click.Path())
def main(ngaqua_root, out_path, t, sam, parameters):
    dt = 30.0
    n = 10

    ic = get_ngqaua_ic(ngaqua_root, t)

    if parameters:
        with open(parameters) as f:
            prm = json.load(f)
    else:
        prm = get_parameters(n, dt)
    path = run_sam_nsteps(ic, prm, sam_src=abspath(sam))
    files = os.path.join(path, 'OUT_3D', '*.nc')

    # open files
    ds = xr.open_mfdataset(files)

    ds = ds.rename({'time': 'step'})
    ds = ds.assign_coords(time=ic.time, step=ds.step - ds.step[0])
    ds = ds.expand_dims('time')
    ds.attrs['sam_namelist'] = json.dumps(prm)
    ds.to_netcdf(out_path, unlimited_dims=['time'], engine='h5netcdf')

    # clean up SAM run
    shutil.rmtree(path)
    return


if __name__ == '__main__':
    main()
