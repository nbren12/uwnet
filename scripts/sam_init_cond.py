#!/usr/bin/env python
"""Run SAM starting from initial conditions
"""
import subprocess
import argparse
import numpy as np
import os
import shutil
import xarray as xr
import tempfile

full_physics = """
 dosgs		= .true.,
 dodamping 	= .true.,
 doupperbound  	= .true.,
 docloud 	= .true.,
 doprecip 	= .true.,
 dolongwave	= .true.,
 doshortwave	= .true.,
 dosurface 	= .true.,
 dolargescale 	= .false.,
 doradforcing   = .false.,
 dosfcforcing   = .false.,
 donudging_uv   = .false.,
 donudging_tq   = .false.,

 doradlat = .true.
 doseasons = .false.
 doequinox = .true.
 docoriolis = .true.
 dofplane = .false.

"""

dry_physics = """
 dosgs		= .true.,
 dodamping 	= .true.,
 doupperbound  	= .true.,
 docloud 	= .false.,
 doprecip 	= .false.,
 dolongwave	= .false.,
 doshortwave	= .false.,
 dosurface 	= .false.,
 dolargescale 	= .false.,
 doradforcing   = .false.,
 dosfcforcing   = .false.,
 donudging_uv   = .false.,
 donudging_tq   = .false.,

 doradlat = .true.
 doseasons = .false.
 doequinox = .true.
 docoriolis = .true.
 dofplane = .false.

"""

rad_physics = """
 dosgs		= .true.,
 dodamping 	= .true.,
 doupperbound  	= .true.,
 docloud 	= .false.,
 doprecip 	= .false.,
 dolongwave	= .true.,
 doshortwave	= .true.,
 dosurface 	= .false.,
 dolargescale 	= .false.,
 doradforcing   = .false.,
 dosfcforcing   = .false.,
 donudging_uv   = .false.,
 donudging_tq   = .false.,

 doradlat = .true.
 doseasons = .false.
 doequinox = .true.
 docoriolis = .true.
 dofplane = .false.

"""

physics_options = {
    'rad': rad_physics,
    'full': full_physics,
    'dry': dry_physics
}

namelist_template = """
 &PARAMETERS

 nrestart = 0

 caseid ='test'
 !caseid_restart ='ngaqua.dt45.QOBS'
 !case_restart = 'NGAqua'

 CEM = .true.,
 OCEAN = .true.,

 {physics}

 ocean_type = 3 !QOBS
 perturb_type = 23
 initial_condition_netcdf = 'NGAqua/ic.nc'

 dx =   160000.,
 dy = 	160000.,
 dt = 	  30.,


 latitude0= 0.72,
 longitude0=0.0,
 nrad = 30,

 day0= {day0}

 nstop 		=  {nstop}, ! 5 days
 nprint 	= {nstop},
 nstat 		= {nstop},
 nstatfrq 	= 1,

 nsave2D	= 360,
 nsave2Dstart	= 99960480,
 nsave2Dend	= 99960480,
 save2Dbin = .true.,

 nsave3D	= {nstop},
 nsave3Dstart	= 0,
 nsave3Dend	= 99960480,
 save3Dbin = .true.,

 doSAMconditionals = .false.
 dosatupdnconditionals = .false.

/ &end


&SGS_TKE

 dosmagor	= .true.,

/ &end

"""

sounding_template = """ z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]
   0.000000              40   1012.22571
  -999.000000  1000.000000   300.125000    0.0     0.0    0.0
  -999.000000   975.000000   300.431488    0.0     0.0    0.0
  -999.000000   950.000000   300.896698    0.0     0.0    0.0
  -999.000000   925.000000   301.765717    0.0     0.0    0.0
  -999.000000   900.000000   302.736877    0.0     0.0    0.0
  -999.000000   875.000000   303.797699    0.0     0.0    0.0
  -999.000000   850.000000   305.123962    0.0     0.0    0.0
  -999.000000   825.000000   306.223816    0.0     0.0    0.0
  -999.000000   800.000000   307.851776    0.0     0.0    0.0
  -999.000000   775.000000   308.928864    0.0     0.0    0.0
  -999.000000   750.000000   310.489685    0.0     0.0    0.0
  -999.000000   725.000000   311.750458    0.0     0.0    0.0
  -999.000000   700.000000   313.074036    0.0     0.0    0.0
  -999.000000   675.000000   314.068695    0.0     0.0    0.0
  -999.000000   650.000000   315.505066    0.0     0.0    0.0
  -999.000000   625.000000   316.961090    0.0     0.0    0.0
  -999.000000   600.000000   318.314911    0.0     0.0    0.0
  -999.000000   575.000000   319.887146    0.0     0.0    0.0
  -999.000000   550.000000   322.109283    0.0     0.0    0.0
  -999.000000   525.000000   324.871643    0.0     0.0    0.0
  -999.000000   500.000000   326.937561    0.0     0.0    0.0
  -999.000000   475.000000   328.604462    0.0     0.0    0.0
  -999.000000   450.000000   330.208527    0.0     0.0    0.0
  -999.000000   425.000000   332.371063    0.0     0.0    0.0
  -999.000000   400.000000   335.075592    0.0     0.0    0.0
  -999.000000   375.000000   337.002319    0.0     0.0    0.0
  -999.000000   350.000000   338.537476    0.0     0.0    0.0
  -999.000000   325.000000   339.830200    0.0     0.0    0.0
  -999.000000   300.000000   341.060333    0.0     0.0    0.0
  -999.000000   275.000000   342.228668    0.0     0.0    0.0
  -999.000000   250.000000   343.384430    0.0     0.0    0.0
  -999.000000   225.000000   344.720947    0.0     0.0    0.0
  -999.000000   200.000000   346.460693    0.0     0.0    0.0
  -999.000000   175.000000   348.576630    0.0     0.0    0.0
  -999.000000   150.000000   352.742554    0.0     0.0    0.0
  -999.000000   125.000000   362.853302    0.0     0.0    0.0
  -999.000000   100.000000   383.417358    0.0     0.0    0.0
  -999.000000    75.000000   421.671814    0.0    -0.0    0.0
  -999.000000    50.000000   488.842316    0.0    -0.0    0.0
  -999.000000    25.000000   608.551453    0.0    -0.0    0.0
   99999.0000              40   1012.22571
  -999.000000  1000.000000   300.125000    0.0     0.0    0.0
  -999.000000   975.000000   300.431488    0.0     0.0    0.0
  -999.000000   950.000000   300.896698    0.0     0.0    0.0
  -999.000000   925.000000   301.765717    0.0     0.0    0.0
  -999.000000   900.000000   302.736877    0.0     0.0    0.0
  -999.000000   875.000000   303.797699    0.0     0.0    0.0
  -999.000000   850.000000   305.123962    0.0     0.0    0.0
  -999.000000   825.000000   306.223816    0.0     0.0    0.0
  -999.000000   800.000000   307.851776    0.0     0.0    0.0
  -999.000000   775.000000   308.928864    0.0     0.0    0.0
  -999.000000   750.000000   310.489685    0.0     0.0    0.0
  -999.000000   725.000000   311.750458    0.0     0.0    0.0
  -999.000000   700.000000   313.074036    0.0     0.0    0.0
  -999.000000   675.000000   314.068695    0.0     0.0    0.0
  -999.000000   650.000000   315.505066    0.0     0.0    0.0
  -999.000000   625.000000   316.961090    0.0     0.0    0.0
  -999.000000   600.000000   318.314911    0.0     0.0    0.0
  -999.000000   575.000000   319.887146    0.0     0.0    0.0
  -999.000000   550.000000   322.109283    0.0     0.0    0.0
  -999.000000   525.000000   324.871643    0.0     0.0    0.0
  -999.000000   500.000000   326.937561    0.0     0.0    0.0
  -999.000000   475.000000   328.604462    0.0     0.0    0.0
  -999.000000   450.000000   330.208527    0.0     0.0    0.0
  -999.000000   425.000000   332.371063    0.0     0.0    0.0
  -999.000000   400.000000   335.075592    0.0     0.0    0.0
  -999.000000   375.000000   337.002319    0.0     0.0    0.0
  -999.000000   350.000000   338.537476    0.0     0.0    0.0
  -999.000000   325.000000   339.830200    0.0     0.0    0.0
  -999.000000   300.000000   341.060333    0.0     0.0    0.0
  -999.000000   275.000000   342.228668    0.0     0.0    0.0
  -999.000000   250.000000   343.384430    0.0     0.0    0.0
  -999.000000   225.000000   344.720947    0.0     0.0    0.0
  -999.000000   200.000000   346.460693    0.0     0.0    0.0
  -999.000000   175.000000   348.576630    0.0     0.0    0.0
  -999.000000   150.000000   352.742554    0.0     0.0    0.0
  -999.000000   125.000000   362.853302    0.0     0.0    0.0
  -999.000000   100.000000   383.417358    0.0     0.0    0.0
  -999.000000    75.000000   421.671814    0.0    -0.0    0.0
  -999.000000    50.000000   488.842316    0.0    -0.0    0.0
  -999.000000    25.000000   608.551453    0.0    -0.0    0.0
"""

def save_snd(path):
    with open(path, "w") as f:
        f.write(sounding_template)


def save_grd(z, path):
    # extrapolate z
    z = np.hstack((z, 2*z[-1] - z[-2]))
    np.savetxt(path, z, fmt="%10.5f")


def save_ic(stag, cent, stat, path):
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

    ic.to_netcdf(path)


def save_namelist(time_interval, day0, physics, path):
    dt = 30.0
    nstop = int(time_interval // dt)
    with open(path, "w") as f:
        namelist = namelist_template.format(day0=day0, nstop=nstop,
                                            physics=physics_options[physics])
        print(namelist)
        f.write(namelist)


def save_rundata(path, src="/Users/noah/workspace/models/SAMUWgh/RUNDATA"):
    shutil.copytree(src, path)


def save_sam_dir(stag, cent, stat, path, time_interval=900.0,
                 physics='full'):
    for out_dir in 'OUT_3D OUT_2D OUT_STAT RESTART NGAqua'.split():
        os.mkdir(os.path.join(path, out_dir))

    # prepare case directory
    case_dir = os.path.join(path, out_dir)
    prm = os.path.join(case_dir, 'prm')
    casename = os.path.join(path, 'CaseName')
    rundata = os.path.join(path, 'RUNDATA')
    snd = os.path.join(case_dir, 'snd')
    grd = os.path.join(case_dir, 'grd')
    nc = os.path.join(case_dir, 'ic.nc')

    day0 = float(stag.time)
    save_namelist(time_interval, day0, physics, prm)
    save_snd(snd)
    save_grd(stag.z, grd)
    save_ic(stag, cent, stat, nc)
    save_rundata(rundata)

    with open(casename, "w") as f:
        f.write("NGAqua")


def run_sam(path, docker_image="nbren12/sam",
            exe="/sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM"):
    path = os.path.abspath(path)
    return subprocess.call(['docker', 'run',
                            "-v", path + ":/run",
                            "-w", "/run",
                            docker_image, exe])


def convert_nc(path, docker_image="nbren12/sam"):
    folder = os.path.dirname(path)
    name = os.path.basename(path)
    folder = os.path.abspath(folder)
    subprocess.call(['docker', 'run',
                     "-v", folder + ":/tmp",
                     "-w", "/tmp",
                     docker_image, 'bin3D2nc', name])
    output_name = os.path.splitext(path)[0] + '.nc'
    return output_name

def combine_steps(ds1, ds2):

    dt = float(ds2.time) - float(ds1.time)

    ds2['time'] = ds1.time
    ds = xr.concat([ds1, ds2], dim='step')
    ds['dt'] = xr.DataArray(dt,
                            attrs={'long_name': 'timestep', 'units': 'day'})

    return ds


def rename_var(z, coords):
    rename_d = {'xc': 'x', 'yc': 'y', 'ys': 'y', 'xs': 'x'}
    rename_d = {key: val for key, val in rename_d.items() if key in z.dims}

    return z.rename(rename_d).assign_coords(**coords)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create initial condition for'
                                     'coarse resolution SAM')

    parser.add_argument('basedir', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-t', '--time', type=int, default=0)
    parser.add_argument('-r', '--retry', type=int, default=10)
    parser.add_argument('-p', '--physics', type=str, default='full',
                        help='Physics package')

    return parser.parse_args()


args = parse_arguments()

time = args.time
stagger_path = os.path.join(args.basedir, "stagger", "3d", "all.nc")
center_path = os.path.join(args.basedir, "coarse", "3d", "all.nc")
stat_path = os.path.join(args.basedir, "stat.nc")

# open necessary files
cent = xr.open_dataset(center_path, engine='netcdf4').isel(time=time)
time = cent.time
stag = (xr.open_dataset(stagger_path).sel(time=time)
        .apply(lambda x: rename_var(x, cent.coords)))
stat = xr.open_dataset(stat_path)

remove_queue = []
# Create SAM directory and run SAM
for _ in range(args.retry):
    output_dir = tempfile.mkdtemp(dir=".", prefix=".tmp")
    remove_queue.append(output_dir)

    save_sam_dir(stag, cent, stat, output_dir, physics=args.physics)
    run_sam(output_dir)

    ncfile1 = convert_nc(f"{output_dir}/OUT_3D/NGAqua_test_1_0000000000.bin3D")
    ncfile2 = convert_nc(f"{output_dir}/OUT_3D/NGAqua_test_1_0000000030.bin3D")
    ds1 = xr.open_dataset(ncfile1)
    try:
        ds2 = xr.open_dataset(ncfile2)
    except OSError:
        print("SAM did not run succesfully. Retrying")
    else:
        break

combine_steps(ds1, ds2).to_netcdf(args.output)

# delete temporary directories
for output_dir in remove_queue:
    shutil.rmtree(output_dir)
