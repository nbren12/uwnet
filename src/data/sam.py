import attr
import f90nml
import xarray as xr
from os.path import join
import glob


@attr.s
class SAMRun(object):
    """Class for opening SAM simulation results"""
    path = attr.ib()
    case = attr.ib(default='')

    def open_ncfiles_in_dir(self, dir, suffix='_'):
        return xr.open_mfdataset(f"{self.path}/{dir}/*_{self.case}{suffix}*.nc")

    @property
    def debug_files(self):
        return glob.glob(join(self.path, "*.pkl"))

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
    def data_2d(self):
        return self.open_ncfiles_in_dir("OUT_2D")

    @property
    def namelist(self):
        return f90nml.read(f"{self.path}/CASE/prm")
