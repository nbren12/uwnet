"""
Energy and moisture are conserved if the following vertically integrated quantities are true:

    cp <Q1> = L_v Prec + SHF + RADSFC - RADTOP
    <Q2> = Evap - Prec

"""
from toolz import curry
import torch


def apply_linear_constraint(lin, a, x, *args, inequality=False, v=None,
                            **kwargs):
    """Apply a linear constraint

    Parameters
    ----------
    lin : Callable
        the linear constraint of the form lin(x, *args, **kwargs) = a that is linear in x
    inequality : bool
        assume that lin(x, *args, **kwargs) >= a

    Returns
    -------
    y : Tensor
      x transformed to satisfy the constraint

    """

    val_x = lin(x, *args, **kwargs)

    # use a constant adjustment
    # x - alpha * v
    if v is None:
        v = torch.ones(x.size(-1))
    val_v = lin(v, *args, **kwargs)
    alpha = (val_x - a)

    if inequality:
        alpha = alpha.clamp(max=0)

    return x - v * (alpha / val_v)


@curry
def expected_moisture(q0, fqt, precip, lhf, h, layer_mass):
    """Same as energy imbalance but for moisture budget

    Parameters
    ----------
    q0  (g/kg)
        moisture before
    fqt (g/kg/day)
        moistening from horizontal advection.
    h   (day)
        time step
    precip (mm/d)
        sfc flux over time step
    lhf  (W/m2)

    Returns
    -------
    current_pw, expected_pw (g/m2)

    """
    Lv = 2.51e6
    density_liquid = 1000.0

    water0 = (q0 * layer_mass).sum(-1, keepdim=True)/1000.0            # kg
    fqt_int = (fqt * layer_mass).sum(-1, keepdim=True)/1000.0 / 86400  # kg/s
    net_evap = lhf / Lv - precip / 1000.0 * density_liquid / 86400     # kg/s
    h = h * 86400  # s
    return water0*1000, 1000 * (water0 + (fqt_int + net_evap) * h)


def expected_temperature(sl, fsl, pw_change, shf, radtoa, radsfc, h,
                         layer_mass):
    """Expected column average SLI to conserve energy

    Parameters
    ----------
    sl (K)
        output before neural network
    fsl (K/day)
        heat transport without surface fluxes
    pw_change (g/m2)
        change in precipitable water over time step
    shf, radtoa, radsfc (W/m2)
        surface and toa fields (upward)
    h (day)
        time step
    layer_mass (kg/m2)

    Returns
    -------
    sl0, sl_int (K / kg / m^2)
        integrals of before/after SLI

    """
    Lv = 2.51e6
    cp = 1004

    sl0 = (layer_mass * sl).sum(-1)
    fsl_int = (layer_mass * fsl).sum(-1)
    h_seconds = h * 86400

    energy_change = pw_change / 1000 * Lv + (shf + radsfc - radtoa) * h_seconds
    sl1 = sl0 + fsl_int * h + energy_change / cp
    return sl0, sl1


def enforce_expected_integral(x, int, layer_mass):
    int_actual = (x * layer_mass).sum(-1)
    return x * int/int_actual


def fix_negative_moisture(q, layer_mass):
    """Fix negative moisture in moisture conserving way"""
    eps = torch.tensor(1e-10, requires_grad=False)

    water_before = (q*layer_mass).sum(-1, keepdim=True)
    q = q.clamp(eps)
    water_after = (q*layer_mass).sum(-1, keepdim=True)
    return q * water_before/water_after


def apply_constraints(x0, x1, time_step):
    """Main function for applying constraints"""

    layer_mass = x0['layer_mass']
    qt = fix_negative_moisture(x1['qt'], x0['layer_mass'])
    if 'Prec' in x0:
        pw0, pw = expected_moisture(x0['qt'], x0['FQT'], x1['Prec'],
                                    x1['LHF'], time_step, x0['layer_mass'])

        sl_int0, sl_int = expected_temperature(x0['sl'], x0['FSL'], pw0-pw,
                                               x1['SHF'], x1['RADTOA'],
                                               x1['RADSFC'], time_step)

        sl = enforce_expected_integral(x1['sl'], sl_int, layer_mass)
        qt = enforce_expected_integral(x1['qt'], pw, layer_mass)
    else:
        sl = x1['sl']

    x1['qt'] = qt
    x1['sl'] = sl
    return x1
