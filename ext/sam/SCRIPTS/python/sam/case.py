"""Objects for configuring and running SAM
"""
import glob
import os
import shutil
import subprocess
import tempfile

import attr
import numpy as np
from toolz import assoc_in

import f90nml
import xarray as xr

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

default_parameters = {
    'parameters': {
        'cem': True,
        'day0': 100.625,
        'docloud': False,
        'docolumn': False,
        'docoriolis': True,
        'dodamping': False,
        'doequinox': True,
        'dofplane': False,
        'dolargescale': False,
        'dolongwave': False,
        'donudging_tq': False,
        'donudging_uv': False,
        'doprecip': False,
        'doradforcing': False,
        'doradlat': True,
        'dosamconditionals': False,
        'dosatupdnconditionals': False,
        'doseasons': False,
        'dosfcforcing': False,
        'dosgs': False,
        'doshortwave': False,
        'dosurface': False,
        'doupperbound': False,
        'dowally': True,
        'dt': 30.0,
        'dx': 160000.0,
        'dy': 160000.0,
        'perturb_type': 23,
        'initial_condition_netcdf': 'NG1/ic.nc',
        'latitude0': 0.72,
        'longitude0': 0.0,
        'nprint': 240,
        'nrad': 30,
        'nrestart': 0,
        'nsave2d': 120,
        'nsave2dend': 99960480,
        'nsave2dstart': 1000000,
        'nsave3d': 120,
        'nsave3dend': 99960480,
        'nsave3dstart': 10000000,
        'nstat': 240,
        'nstatfrq': 240,
        'nstop': 480,
        'ocean': True,
        'ocean_type': 3,
        'save2dbin': True,
        'save3dbin': True,
    },
    'sgs_tke': {
        'dosmagor': True
    },
    'python': {
        'dopython': False,
        'usepython': False,
        'npython': 1,
        'function_name': 'call_neural_network',
        'module_name': 'uwnet.sam_interface'
    }
}

default_grd = np.array([
    37., 112., 194., 288., 395., 520., 667., 843., 1062., 1331., 1664., 2274.,
    3097., 4119., 5310., 6555., 7763., 8931., 10048., 11116., 12141., 13138.,
    14115., 15063., 15984., 16900., 17800., 18700., 19800., 21000., 22500.,
    24000., 25500., 27000., 28500.
])


def make_docker_cmd(image, exe, **kwargs):
    """Create command list to pass to subprocess

    Parameters
    ----------
    image : str
        Name of docker image
    exe : str
        path to executable in docker
    workdir : str, optional
        working directory in docker container
    volumes : list of tuples, optional
    env : dictionary, optional

    Returns
    -------
    cmd : list
        list of arguments which can be passed to subprocess
    """
    cmd = ['docker', 'run']

    # add workdir
    try:
        cmd += ['-w', kwargs['workdir']]
    except KeyError:
        pass

    # add volumes
    for src, dest in kwargs.get('volumes', []):
        cmd += ['-v', os.path.abspath(src) + ':' + dest]

    # add environmental variables
    for name, value in kwargs.get('env', []):
        cmd += ['-e', name + '=' + value]

    # add image and executable names
    cmd += [image, exe]

    # finally call
    return cmd


@attr.s
class Case(object):
    """A configuration of the System for Atmospheric Modeling,
    """
    z = attr.ib(default=None)
    name = attr.ib(default='CASE')
    sam_src = attr.ib(default='/sam')
    prm = attr.ib(default=default_parameters)
    exe = attr.ib(default="/sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM")
    path = attr.ib(factory=tempfile.mkdtemp, converter=os.path.abspath)
    docker_image = attr.ib(default="nbren12/samuwgh:latest")
    env = attr.ib(factory=dict)

    @property
    def _z(self):
        return self.z

    def save(self):

        path = self.path

        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        for out_dir in f'OUT_3D OUT_2D OUT_STAT RESTART {self.name}'.split():
            os.mkdir(os.path.join(path, out_dir))

        # get path names
        case_dir = os.path.join(path, self.name)
        prm = os.path.join(case_dir, 'prm')
        casename = os.path.join(path, 'CaseName')
        rundata = os.path.join(path, 'RUNDATA')
        snd = os.path.join(case_dir, 'snd')
        grd = os.path.join(case_dir, 'grd')

        self.save_prm(prm)
        self.save_grd(self._z, grd)
        self.save_snd(snd)
        try:
            self.save_rundata(rundata)
        except FileNotFoundError:
            pass

        with open(casename, "w") as f:
            f.write(self.name)

    def save_prm(self, path):
        f90nml.write(self.get_prm(), path)

    @staticmethod
    def save_grd(z, path):
        # extrapolate z
        z = np.hstack((z, 2 * z[-1] - z[-2]))
        np.savetxt(path, z, fmt="%10.5f")

    @staticmethod
    def save_snd(path):
        with open(path, "w") as f:
            f.write(sounding_template)

    def save_rundata(self, path):
        src = f"{self.sam_src}/RUNDATA"
        shutil.copytree(src, path)

    def run(self):
        self.save()
        subprocess.call([self.exe], cwd=self.path)

    def run_docker(self):
        self.save()

        cmd = make_docker_cmd(image=self.docker_image,
                              exe=self.exe,
                              volumes=[(self.path, '/run')],
                              workdir="/run")
        return subprocess.call(cmd)

    def convert_files_to_netcdf(self):
        cmd = make_docker_cmd(image=self.docker_image,
                              exe='/sam/docker/convert_files.sh',
                              workdir="/run",
                              volumes=[(self.path, '/run')])
        return subprocess.call(cmd)

    def get_prm(self):
        return self.prm


@attr.s
class InitialConditionCase(Case):
    ic = attr.ib(default=None)

    @property
    def _z(self):
        return self.ic.z.values

    @property
    def initial_condition_path(self):
        return os.path.join(self.path, self.name, 'ic.nc')

    def save_ic(self):
        self.ic.to_netcdf(self.initial_condition_path)

    def save(self):
        super(InitialConditionCase, self).save()
        self.save_ic()

    def get_prm(self):
        path = os.path.relpath(self.initial_condition_path, self.path)
        self.prm['parameters']['initial_condition_netcdf'] = path


def _rename_var(z, coords):
    rename_d = {'xc': 'x', 'yc': 'y', 'ys': 'y', 'xs': 'x'}
    rename_d = {key: val for key, val in rename_d.items() if key in z.dims}

    return z.rename(rename_d).assign_coords(**coords)


def get_ngqaua_ic(basedir, time):
    """Get an initial condition for time t from the NGAQUA"""
    stagger_path = os.path.join(basedir, "stagger", "3d", "all.nc")
    center_path = os.path.join(basedir, "coarse", "3d", "all.nc")
    stat_path = os.path.join(basedir, "stat.nc")

    # open necessary files
    cent = xr.open_dataset(center_path, engine='netcdf4').isel(time=time)
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


def pressure_correct(ic, path=None, sam_src="."):
    """Pressure correct the velocity fields using SAM"""

    if path is None:
        path = tempfile.mkdtemp(dir=".")

    prm = default_parameters

    for key, val in [('nstop', 0),
                     ('nsave3d', 1), ('nstat', 0), ('nstatfrq', 1),
                     ('dt', .0001), ('nsave3dstart', 0)]:
        prm = assoc_in(prm, ['parameters', key], val)

    case = InitialConditionCase(ic=ic, path=path, sam_src=sam_src, prm=prm)
    case.run_docker()
    case.convert_files_to_netcdf()

    files = glob.glob(os.path.join(case.path, 'OUT_3D', '*.nc'))

    ic = ic.drop('time')
    ds = xr.open_dataset(files[-1]).load()\
                                   .assign_coords(x=ic.x, y=ic.y)
    shutil.rmtree(path)

    return ds
