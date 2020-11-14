"""Thermodynamic and other math calculations
"""
from functools import partial
import numpy as np
import xarray as xr
from .xcalc import centderiv

grav = 9.81
R = 287.058
cp = 1004
kappa = R / cp
Lc = 2.5104e6
rho0 = 1.19
sec_in_day = 86400
liquid_water_density = 1000.0

rad_earth = 6371e3  # m
circumference_earth = rad_earth * 2 * np.pi


def compute_insolation(lat, day, scon=1367, eccf=1.0):
    """Compute the solar insolation in W/m2 assuming perpetual equinox

    Parameters
    ----------
    lat : (ny, nx)
        latitude in degrees
    day : float
        day of year. Only uses time of day (the fraction).
    scon : float
        solar constant. Default 1367 W/m2
        eccentricity factor. Ratio of orbital radius at perihelion and
        aphelion. Default 1.0.

    """
    time_of_day = day % 1.0

    # cos zenith angle
    mu = -np.cos(2 * np.pi * time_of_day) * np.cos(np.pi * lat / 180)
    mu[mu < 0] = 0.0
    return scon * eccf * mu

def metpy_wrapper(fun):
    """Given a metpy function return an xarray compatible version
    """
    from metpy.units import units as u

    def func(*args):
        def f(*largs):
            new_args = [
                u.Quantity(larg, arg.units) for larg, arg in zip(largs, args)
            ]
            return fun(*new_args)

        output_units = f(* [1 for arg in args]).units
        ds = xr.apply_ufunc(f, *args)
        ds.attrs['units'] = str(output_units)
        return ds

    return func


def omega_from_w(w, rho):
    """Presure velocity in anelastic framework

    omega = dp_0/dt = dp_0/dz dz/dt = - rho_0 g w
    """
    return -w * rho * grav


def liquid_water_temperature(t, qn, qp):
    """This is an approximate calculation neglecting ice and snow
    """

    sl = t + grav / cp * t.z - Lc / cp * (qp + qn) / 1000.0
    sl.attrs['units'] = 'K'
    return sl


def total_water(qv, qn):
    qt = qv + qn
    qt.attrs['units'] = 'g/kg'
    return qt


def get_dz(z):
    zext = np.hstack((-z[0], z, 2.0 * z[-1] - 1.0 * z[-2]))
    zw = .5 * (zext[1:] + zext[:-1])
    dz = zw[1:] - zw[:-1]

    return xr.DataArray(dz, z.coords)


def interface_heights(z):
    zext = np.hstack((-z[0], z, 2.0 * z[-1] - 1.0 * z[-2]))
    return .5 * (zext[1:] + zext[:-1])


def layer_mass(rho):
    dz = get_dz(rho.z)
    return (rho * dz).assign_attrs(units='kg/m2')


def layer_mass_from_p(p, ps=None):
    if ps is None:
        ps = 2 * p[0] - p[1]

    ptop = p[-1] * 2 - p[-2]

    pext = np.hstack((ps, p, ptop))
    pint = (pext[1:] + pext[:-1]) / 2
    dp = -np.diff(pint * 100) / grav

    return xr.DataArray(dp, p.coords)


def mass_integrate(p, x, average=False):
    dp = layer_mass_from_p(p)
    ans = (x * dp).sum(p.dims)

    if average:
        ans = ans / dp.sum()

    return ans


def column_rh(QV, TABS, p):
    from metpy.calc import relative_humidity_from_mixing_ratio
    rh = metpy_wrapper(relative_humidity_from_mixing_ratio)(QV, TABS, p)

    return mass_integrate(p, rh / 1000, average=True)


def ngaqua_y_to_lat(y, y0=5120000):
    rad_earth = 6371e3  # m
    circumference_earth = rad_earth * 2 * np.pi
    return (y - y0) / circumference_earth * 360


def coriolis_ngaqua(y):
    lat = ngaqua_y_to_lat(y)
    omega = 2 * np.pi / 86400
    return 2 * omega * np.sin(np.deg2rad(lat))


def get_geostrophic_winds(p, rho, min_cor=1e-5):
    """Compute geostrophic winds

    Parameters
    ----------
    p : xr.DataArray
        partial pressure in (Pa)
    rho : xr.DataArray
        density in (kg/m3)
    min_cor : float
        minimum coriolis paramter

    Returns
    -------
    ug, vg : xr.DataArray
        the geostropohc wind fields, with values masked for locations where the
        absolute coriolis parameter is smaller than min_cor.

    """
    # get coriolis force
    fcor = coriolis_ngaqua(p.y)
    px = centderiv(p, dim='x') / rho
    py = centderiv(p, dim='y') / rho

    vg = px / fcor
    vg = vg.where(np.abs(fcor) > min_cor)
    vg.name = "VG"

    ug = -py / fcor
    ug = ug.where(np.abs(fcor) > min_cor)
    ug.name = "UG"
    return ug, vg


def compute_apparent_source(prog, forcing):
    dt = prog.time[1] - prog.time[0]
    avg_forcing = (forcing + forcing.shift(time=-1)) / 2
    return (prog.shift(time=-1) - prog) / dt - avg_forcing


def compute_q2(ngaqua):
    return compute_apparent_source(ngaqua.QT, ngaqua.FQT * 86400)


def vorticity(u, v):
    psi = u.differentiate('y') - v.differentiate('x')
    psi.name = 'Vorticity'
    return psi


def lhf_to_evap(lhf):
    rhow = 1000
    evap = lhf / 2.51e6 / rhow * 86400 * 1000
    evap.name = 'Evaporation'
    evap.attrs['units'] = 'mm/day'
    return evap


def integrate_q1(q1, layer_mass, dim='z'):
    """Vertically integrate Q1 (K/day) to give W/m2
    """
    return (q1 * layer_mass).sum(dim) * (cp / sec_in_day)


def integrate_q2(q2, layer_mass, dim='z'):
    """Vertically integrate Q2 (g/kg/day) to give mm/day
    """
    return (q2 * layer_mass).sum(dim) / liquid_water_density


def net_precipitation_from_training(data):
    """Compute Net Precipitation from Q2

    This is not exactly equivalent to precipitation minus evaporation due to
    sampling issue.
    """
    return -integrate_q2(
        compute_apparent_source(data.QT, data.FQT * 86400), data.layer_mass)


def net_precipitation_from_prec_evap(data):
    return data.Prec - lhf_to_evap(data.LHF)


def net_heating(prec, shf, swns, swnt, lwns, lwnt):
    surface_radiation_net_upward = (lwns - swns)
    toa_radiation_net_upward = (lwnt - swnt)
    net_radiation = surface_radiation_net_upward - toa_radiation_net_upward

    return prec * (Lc / sec_in_day) + net_radiation


def net_heating_from_data_2d(data_2d):
    prec = data_2d.Prec
    shf = data_2d.SHF
    swns = data_2d.SWNS
    swnt = data_2d.SWNT
    lwns = data_2d.LWNS
    lwnt = data_2d.LWNT
    return net_heating(prec, shf, swns, swnt, lwns, lwnt)


def periodogram(pw: xr.DataArray, dim='x', freq_name='f'):
    from scipy import signal
    axis = pw.get_axis_num(dim)

    x = pw.values

    coord = pw[dim]
    d = float(coord[1]-coord[0])
    f, x = signal.periodogram(x, axis=axis, fs=1/d)

    dims = list(pw.dims)
    dims[pw.get_axis_num(dim)] = freq_name
    coords = {key: pw[key] for key in pw.dims if key != dim}
    coords[freq_name] = f

    return xr.DataArray(x, dims=dims, coords=coords)


def water_budget(data_2d):
    """Compute precipitable water budget from 2D data"""
    storage = data_2d.PW.differentiate('time')
    advection = storage + data_2d.NPNN
    return xr.Dataset({'storage': storage, 'advection': advection, 'net_precip':  data_2d.NPNN})


def potential_temperature(temperature_kelvin, pressure_mb, p0=1015.0):
    return temperature_kelvin * (p0 / pressure_mb)**kappa


def lower_tropospheric_stability(
        temperature_kelvin, pressure_mb, sst, p0=1015.0):
    theta = potential_temperature(temperature_kelvin, pressure_mb, p0)
    i = int(np.argmin(np.abs(np.asarray(pressure_mb) - 700)))
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


def omega_from_w(w, rho):
    """Presure velocity in anelastic framework

    omega = dp_0/dt = dp_0/dz dz/dt = - rho_0 g w
    """
    return -w * rho * grav