import attr
import f90nml
import xarray as xr
import glob


@attr.s
class SAMRun(object):
    """Class for opening SAM simulation results"""
    path = attr.ib()
    case = attr.ib(default='')
    file_limit = attr.ib(default=400)

    def list_ncfiles_in_dir(self, dir, suffix='_'):
        pattern = f"{self.path}/{dir}/*_{self.case}{suffix}*.nc"
        files = glob.glob(pattern)
        return files[:min(len(files), self.file_limit)]

    def open_ncfiles_in_dir(self, *args, **kwargs):
        files = self.list_ncfiles_in_dir(*args, **kwargs)
        return xr.open_mfdataset(files)

    @property
    def debug_files(self):
        suffix = '_'
        ext = 'pt'
        pattern = f"{self.path}/OUT_3D/*_{self.case}{suffix}*.{ext}"
        return glob.glob(pattern)

    def open_debug(self, i):
        import torch
        return torch.load(self.debug_files[i])

    @property
    def stat(self):
        return self.open_ncfiles_in_dir("OUT_STAT", suffix='')

    @property
    def data_3d(self):
        return self.open_ncfiles_in_dir("OUT_3D")

    @property
    def files_3d(self):
        return self.list_ncfiles_in_dir('OUT_3D')

    @property
    def data_2d(self):
        return self.open_ncfiles_in_dir("OUT_2D")

    @property
    def namelist(self):
        return f90nml.read(f"{self.path}/CASE/prm")


def state_to_xarray(state):
    """Convert output from debug file to xarray"""
    time = state['day']
    dims = state['_DIMS']
    dims['layer_mass'] = 'mid_levels'
    dims['tendency_of_total_water_mixing_ratio_due_to_neural_network'] = dims[
        'total_water_mixing_ratio']

    data_vars = {}
    for key in dims:
        arr = state[key]
        ndim = len(dims[key])
        if ndim == 2:
            arr = arr[0]
        data_vars[key] = (dims[key], arr)

    ds = xr.Dataset(data_vars)
    ds['time'] = time
    return ds
