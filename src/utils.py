import numpy as np
import xarray as xr


def get_regions(y):
    """Get the tropics subtropics and extratropics mask

    This is especially useful for groupby operations.
    """
    tropics_bndy = .25
    subtropics_north_bndy = .50
    dx = 160e3

    percent = 2 * y / (y.max() + dx) - 1

    subtropics = ((np.abs(percent) > tropics_bndy) &
                  (np.abs(percent) <= subtropics_north_bndy))

    tropics = np.abs(percent) <= tropics_bndy
    return xr.where(tropics, 'Tropics',
                    xr.where(subtropics, 'Subtropics', 'Extratropics'))
