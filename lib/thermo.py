import numpy as np
import xarray as xr

from .util import wrap_xarray_calculation

grav = 9.81
cp = 1004
Lc = 2.5104e6
rho0 = 1.19


def omega_from_w(w, rho):
    """Presure velocity in anelastic framework

    omega = dp_0/dt = dp_0/dz dz/dt = - rho_0 g w
    """
    return - w * rho * grav


def q2(d):
    out = -d.f_qt * Lc/cp / 1000.0
    out.attrs['units'] = 'K/d'
    return out.to_dataset(name='Q2')


def q1(d):
    out = d.f_sl
    out.attrs['units'] = 'K/d'
    return out.to_dataset(name='Q1')


@wrap_xarray_calculation
def liquid_water_temperature(t, qn, qp):
    """This is an approximate calculation neglecting ice and snow
    """

    sl = t + grav/cp * t.z - Lc/cp * (qp + qn)/1000.0
    sl.attrs['units'] = 'K'
    return sl


def d_liquid_water_temperature(divt, divqn, divqp):
    sl = divt - Lc/cp * (divqp + divqn)/1000.0
    sl.attrs['units'] = 'K/s'

    return sl


@wrap_xarray_calculation
def total_water(qv, qn):
    qt = qv + qn
    qt.attrs['units'] = 'g/kg'
    return qt


def get_dz(z):
    zext = np.hstack((-z[0],  z,  2.0*z[-1] - 1.0*z[-2]))
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


def column_rh(QV, TABS, p):
    from metpy.calc import relative_humidity_from_mixing_ratio
    rh = metpy_wrapper(relative_humidity_from_mixing_ratio)(QV, TABS, p)

    return mass_integrate(p, rh/1000, average=True)


def precip_from_ds(dsl, qrad, shf, p):
    return (cp*mass_integrate(p, dsl) - cp*mass_integrate(p, qrad)
            - shf*86400)/Lc


def precip_from_dq(dq, lhf, p):
    return - mass_integrate(p, dq/1000) + lhf/Lc*86400


def mse(sl, qt):
    return cp * sl + qt/1000 * Lc


def shf_to_tendency(shf, rho0):
    rhodz = float(rho0.z * 2 * rho0)
    return shf/cp*86400/rhodz


def lhf_to_tendency(lhf, rho0):
    rhodz = float(rho0.z * 2 * rho0)
    return lhf/Lc*86400/rhodz
