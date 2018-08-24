import base64
import io
import os
import subprocess
import tarfile

import attr
import click
import numpy as np
import zarr
from dask.diagnostics import ProgressBar

import xarray as xr


@attr.s
class ZarrLogger(object):
    path = attr.ib()

    @property
    def root(self):
        return zarr.open_group(self.path, mode="a")

    def append(self, key, val):

        isscalar = val.shape == (1, )

        try:
            arr = self.root[key]
        except KeyError:
            # initialize the group
            if isscalar:
                shape = (0, )
                chunks = (1, )
            else:
                shape = (0, ) + val.shape
                chunks = (1, ) + val.shape

            arr = self.root.zeros(key, shape=shape, chunks=chunks)

        if not isscalar:
            val = val[np.newaxis]

        arr.append(val)

    def append_all(self, x):
        for key in x:
            self.append(key, x[key])

    def set(self, key, val):
        try:
            val.ndim
        except AttributeError:
            val = np.array([val])

        arr = self.root.zeros(key, shape=val.shape)
        arr[:] = val

    def get(self, key):
        return self.root[key]

    def put_attrs(self, key, d):
        return self.root[key].attrs.put(d)

    def set_dims(self, key, dims):
        self.root[key].attrs['_ARRAY_DIMENSIONS'] = dims


def get_var(field, dims, root):
    def get_dim(dim):
        try:
            return root[dim][:]
        except KeyError:
            return

    coords = {dim: root[dim][:] if dim in root else None for dim in dims}
    return xr.DataArray(root[field], dims=dims, coords=coords, name=field)


def get_var3d(field, root):
    x = get_var(field, ['day', 'p', 'y', 'x'], root)
    return x


def get_var2d(field, root):
    x = get_var(field, ['day', 'm', 'y', 'x'], root)
    return x.isel(m=0)


def zarr_to_xr(root):

    vars_3d = ['U', 'V', 'W', 'qt', 'sl', 'FQT', 'FSL', 'Q1NN', 'Q2NN']
    vars_2d = ['Prec', 'LHF']

    data_vars = {}
    for f in vars_3d:
        data_vars[f] = get_var3d(f, root)

    for f in vars_2d:
        data_vars[f] = get_var2d(f, root)

    data_vars['layer_mass'] = get_var('layer_mass', ['p'], root)

    return xr.Dataset(data_vars)


def open_debug_zarr_as_xr(path):
    """Open debugging zarr with and convert to an xarray dataset"""
    ds = zarr.open_group(path)
    return zarr_to_xr(ds)


def get_tar_data(path):
    """Tar a path into a base64 string"""
    tardata = io.BytesIO()
    tar = tarfile.open(fileobj=tardata, mode="w")
    tar.add(path, arcname="./")
    return base64.b64encode(tardata.getvalue()).decode()


def extract_tar_data(s, path):
    """Extract a base64 string from :func:`get_tar_data` to a path"""
    b = base64.b64decode(s)
    buf = io.BytesIO(b)
    f = tarfile.open(fileobj=buf)
    f.extractall(path=path)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse',
                                    'HEAD']).decode().strip()


def get_environ():
    return {key: val for key, val in os.environ.items()}


def get_metadata(state):
    case = state['case']


    # tar the data
    attrs = {
        'case': state['case'],
        'caseid': state['caseid'],
        'folder': get_tar_data(state['case']),
        'env': get_environ()
    }

    try:
        attrs['git'] = {'rev': get_git_revision_hash()}
    except subprocess.CalledProcessError:
        pass

    return attrs


@click.command()
@click.argument('debug_path')
@click.argument('data_path')
@click.argument('output')
def debug_zarr_to_training(debug_path, data_path, output):
    """Prepare training dataset"""

    forcing = xr.open_zarr(debug_path)
    ds = xr.open_zarr(data_path)

    # patch in x, y and z
    forcing['x'] = ds['x']
    forcing['y'] = ds['y']
    forcing['z'] = ds['z']

    # merge the dataset
    print("Number of time points with forcings:", len(forcing.time))

    # plug in forcing information from debugging output
    # this needs to be g/kg/day or K/day
    ds['FQT'] = 86400 * forcing.FQT.shift(time=-1) * 1000
    ds['FSL'] = 86400 * forcing.FSL.shift(time=-1)
    ds = ds.fillna(0.0)

    # remove encoding
    # needed before I can saved the rechunked data
    print("Loading data into memory")
    with ProgressBar():
        ds = ds.chunk({'time': -1, 'x': -1, 'y': -1}).load()

    for key in ds:
        ds[key].encoding = {}

    print("Begin saving data to disk")
    ds.to_zarr(output)

    # add attrs into output
    # I did this using zarr directly because xarray cannot handle json data
    g = zarr.open_group(output)
    g.attrs.put(forcing.attrs)


if __name__ == '__main__':
    debug_zarr_to_training()
