import numpy as np
import xarray as xr

grav = 9.81
cp = 1004
Lc = 2.5104e6
rho0 = 1.19


def metpy_wrapper(fun):
    """Given a metpy function return an xarray compatible version
    """
    from metpy.units import units as u

    def func(*args):
        def f(*largs):
            new_args = [u.Quantity(larg, arg.units)
                        for larg, arg in zip(largs, args)]
            return fun(*new_args)

        output_units = f(*[1 for arg in args]).units
        ds = xr.apply_ufunc(f, *args)
        ds.attrs['units'] = str(output_units)
        return ds

    return func


def omega_from_w(w, rho):
    """Presure velocity in anelastic framework

    omega = dp_0/dt = dp_0/dz dz/dt = - rho_0 g w
    """
    return - w * rho * grav


def liquid_water_temperature(t, qn, qp):
    """This is an approximate calculation neglecting ice and snow
    """

    sl = t + grav/cp * t.z - Lc/cp * (qp + qn)/1000.0
    sl.attrs['units'] = 'K'
    return sl


def total_water(qv, qn):
    qt = qv + qn
    qt.attrs['units'] = 'g/kg'
    return qt


def get_dz(z):
    zext = np.hstack((-z[0], z, 2.0*z[-1] - 1.0*z[-2]))
    zw = .5 * (zext[1:] + zext[:-1])
    dz = zw[1:] - zw[:-1]

    return xr.DataArray(dz, z.coords)


def layer_mass(rho):
    dz = get_dz(rho.z)
    return (rho*dz).assign_attrs(units='kg/m2')


def layer_mass_from_p(p, ps=None):
    if ps is None:
        ps = 2*p[0] - p[1]

    ptop = p[-1]*2 - p[-2]

    pext = np.hstack((ps, p, ptop))
    pint = (pext[1:] + pext[:-1])/2
    dp = - np.diff(pint*100)/grav

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

    return mass_integrate(p, rh/1000, average=True)


def ngaqua_y_to_lat(y, y0=5120000):
    rad_earth = 6371e3  # m
    circumference_earth = rad_earth * 2 * np.pi
    return (y-y0)/circumference_earth * 360


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
    from gnl.xarray import centderiv
    fcor = coriolis_ngaqua(p.y)
    px = centderiv(p, dim='x')/rho
    py = centderiv(p, dim='y')/rho

    vg = px/fcor
    vg = vg.where(np.abs(fcor) > min_cor)
    vg.name = "VG"

    ug = -py/fcor
    ug = ug.where(np.abs(fcor) > min_cor)
    ug.name="UG"
    return ug, vg


def compute_apparent_source(prog, forcing):
    dt = prog.time[1] - prog.time[0]
    avg_forcing = (forcing + forcing.shift(time=-1))/2
    return (prog.shift(time=-1) - prog)/dt - avg_forcing


def compute_q2(ngaqua):
    return compute_apparent_source(ngaqua.QT, ngaqua.FQT*86400)


def vorcitity(u, v):
    f = coriolis_ngaqua(u.y)
    psi = u.differentiate('y') - v.differentiate('x')
    psi.name = 'Vorticity'
    return psi


def lhf_to_evap(lhf):
    rhow = 1000
    evap = lhf / 2.51e6 / rhow * 86400 * 1000
    evap.name = 'Evaporation'
    evap.attrs['units'] = 'mm/day'
    return evap
