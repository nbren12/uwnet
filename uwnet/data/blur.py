import xarray as xr

# needed for arr.ndimage xarray acessor
from .. import ndimage_xarray


def blur(arr, sigma):
    if not {'x', 'y'} <= set(arr.dims):
        raise ValueError
    if sigma <= 1e-8:
        raise ValueError

    return (arr.ndimage.gaussian_filter1d('x', sigma, mode='wrap')
            .ndimage.gaussian_filter1d('y', sigma, mode='nearest'))


def blur_dataset(data, sigma):
    data_vars = {}
    for x in data:
        try:
            data_vars[x] = blur(data[x], sigma)
        except ValueError:
            data_vars[x] = data[x]

    return xr.Dataset(data_vars)
