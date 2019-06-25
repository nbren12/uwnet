import xarray as xr
import json
import argparse
from src.sam.case import get_ngqaua_ic
from uwnet.data.blur import blur_dataset
from uwnet.thermo import layer_mass
from src.data.ngaqua import NGAqua
from src.sam.process_ngaqua import run_sam_nsteps
from subprocess import run
import os
import shutil
from os.path import abspath


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_args_snakemake():
    print("Using snakemake")
    sigma = snakemake.params.sigma
    if sigma:
        sigma = float(sigma)

    return Namespace(
        time_step=int(snakemake.wildcards.step),
        sam_parameters=snakemake.input.sam_parameters,
        sam=snakemake.config.get('sam_path',  '/opt/sam'),
        ngaqua_root=snakemake.params.ngaqua_root,
        output=snakemake.output[0],
        sigma=sigma)


def get_args_argparse():

    parser = argparse.ArgumentParser(
        description='Pre-process a single time step')
    parser.add_argument('-n', '--ngaqua-root', type=str)
    parser.add_argument('-s', '--sam', type=str, default='/opt/sam')
    parser.add_argument('-t', '--time-step', type=int, default=0)
    parser.add_argument('-p', '--sam-parameters', type=str, default=0)
    parser.add_argument('--sigma', type=float, help='Radius for Gaussian blurring')

    parser.add_argument('output')
    return parser.parse_args()


def get_args():
    """Get the arguments needed to run this script

    This will function will behave differently if run from snakemake"""

    try:
        snakemake
    except NameError:
        return get_args_argparse()
    else:
        return get_args_snakemake()


args = get_args()


def get_extra_features(ngaqua: NGAqua, time_step):

    # compute layer_mass
    stat = ngaqua.stat
    rho = stat.RHO.isel(time=0).drop('time')
    w = layer_mass(rho)

    # get 2D variables
    time = ngaqua.times_3d[time_step]
    d2 = ngaqua.data_2d.sel(time=time)

    # add variables to three-d
    d2['RADTOA'] = d2.LWNT - d2.SWNT
    d2['RADSFC'] = d2.LWNS - d2.SWNS
    d2['layer_mass'] = w
    d2['rho'] = rho

    # 3d variables
    qrad = ngaqua.data_3d.QRAD.drop(['x', 'y'])
    d2['QRAD'] = qrad.sel(time=time)

    return d2


# get initial condition
ic = get_ngqaua_ic(args.ngaqua_root, args.time_step)

# get data
ngaqua = NGAqua(args.ngaqua_root)
features = get_extra_features(ngaqua, args.time_step)

if args.sigma:
    print(f"Blurring data with radius {args.sigma}")
    ic = blur_dataset(ic, args.sigma)
    features = blur_dataset(features, args.sigma)

# compute the forcings by running through sam
if args.sam_parameters:
    with open(args.sam_parameters) as f:
        prm = json.load(f)

path = run_sam_nsteps(ic, prm, sam_src=abspath(args.sam))
files = os.path.join(path, 'OUT_3D', '*.nc')
ds = xr.open_mfdataset(files)
shutil.rmtree(path)

for key in ['QT', 'SLI', 'U', 'V']:
    forcing_key = 'F' + key
    src = ds[key].diff('time') / ds.time.diff('time') / 86400
    src = src.isel(time=0)
    ds[forcing_key] = src

ds = ds.isel(time=0)

# forcing data
ic['x'] = ds.x
ic['y'] = ds.y
for key in ic.data_vars:
    if key not in ds:
        ds[key] = ic[key]

ds = ds.merge(features).expand_dims('time')
ds.attrs['sam_namelist'] = json.dumps(prm)
ds.to_netcdf(args.output, unlimited_dims=['time'], engine='h5netcdf')
