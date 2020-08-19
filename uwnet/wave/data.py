import fsspec
import xarray as xr
from .thermo import compute_apparent_source

TRAINING_DATA_PATH = "gs://vcm-ml-data/project_data/uwnet/2020-02-01-noBlur.zarr/"


def open_data(tag):
    if tag == 'training':
        return xr.open_dataset("../../data/processed/training/noBlur.nc")
        return xr.open_zarr(fsspec.get_mapper(TRAINING_DATA_PATH))


def assign_apparent_sources(ds):
    return ds.assign(
        Q1=compute_apparent_source(ds.SLI, 86400 * ds.FSLI),
        Q2=compute_apparent_source(ds.QT, 86400 * ds.FQT))
