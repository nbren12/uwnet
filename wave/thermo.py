from functools import partial

import xarray as xr
import numpy as np

grav = 9.81
R = 287.058
cp = 1004
kappa = R/cp
Lc = 2.5104e6
rho0 = 1.19
sec_in_day = 86400
liquid_water_density = 1000.0

rad_earth = 6371e3  # m
circumference_earth = rad_earth * 2 * np.pi


def ngaqua_y_to_lat(y, y0=5040000):
    return (y - y0) / circumference_earth * 360


def compute_apparent_source(prog, forcing):
    dt = prog.time[1] - prog.time[0]
    avg_forcing = (forcing + forcing.shift(time=-1)) / 2
    return (prog.shift(time=-1) - prog) / dt - avg_forcing


def interface_heights(z):
    zext = np.hstack((-z[0], z, 2.0 * z[-1] - 1.0 * z[-2]))
    return .5 * (zext[1:] + zext[:-1])


def potential_temperature(temperature_kelvin, pressure_mb, p0=1015.0):
    return temperature_kelvin * (p0 / pressure_mb)**kappa


def lower_tropospheric_stability(
        temperature_kelvin, pressure_mb, sst, p0=1015.0):
    theta = potential_temperature(temperature_kelvin, pressure_mb, p0)
    i = np.argmin(np.abs(pressure_mb - 700))
    return theta.isel(z=i) - sst


def water_vapor_path(qv, p, bottom=850, top=550, dim='z'):
    """Water vapor path between specified pressure levels

    Parameters
    ----------
    qv
        water vapor in g/kg
    p
        pressure in mb
    bottom, top : float
        pressure at bottom and top
    dim : default 'z'
        vertical dimension of the data

    Returns
    -------
    path
        water vapor path in mm (liquid water equivalent)

    """

    dp = layer_mass_from_p(p)
    masked = qv.where((p < bottom) & (p > top), 0)
    mass = (masked * dp).sum(dim=dim)
    return mass/liquid_water_density


def layer_mass_from_p(p, ps=None):
    if ps is None:
        ps = 2 * p[0] - p[1]

    ptop = p[-1] * 2 - p[-2]

    pext = np.hstack((ps, p, ptop))
    pint = (pext[1:] + pext[:-1]) / 2
    dp = -np.diff(pint * 100) / grav

    return xr.DataArray(dp, p.coords)


midtropospheric_moisture = partial(water_vapor_path, bottom=850, top=600)
