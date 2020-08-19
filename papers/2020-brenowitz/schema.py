"""Functions for validating binned input data
"""
import xarray as xr

COORDS = ['z', 'path_bins', 'lts_bins']
VERTICAL_VARIABLES = ['QT', 'SLI', 'QT', 'Q2', 'Q1NN', 'Q2NN']


def validate(ds):
    for variable in VERTICAL_VARIABLES:
        dims = ds[variable].dims

        if set(dims) != set(COORDS):
            raise ValueError("Coordinates of the input data are incorrect")

if __name__ == '__main__':
    import sys
    ds = xr.open_dataset(sys.argv[1])
    validate(ds)
