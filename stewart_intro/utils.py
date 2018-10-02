import xarray as xr

project_dir = '~/projects/uwnet/'


def load_data():
    file_location = project_dir + '../2018-09-18-NG_5120x2560x34_4km_10s_QOBS_EQX-SAM_Processed.nc'  # noqa
    return xr.open_dataset(file_location)
