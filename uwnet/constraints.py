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


def energy_imbalance(FSL, shf, prec, radsfc, radtop, layer_mass):
    """Enforce energy and moisture conservation

    Parameters
    ----------
    FSL
        Total tendency from neural network and model
    shf
        sensible heat flux (W/m2)
    prec
        precipitation (m/s)
    radsfc, radtop
        radiative fluxes (upwelling) (W/m2)
    layer_mass
        mass of each vertical level (kg/m2)

    Returns
    -------
    imbalance : torch.tensor
        imbalance from energy conservation

    Notes
    -----
    Energy and moisture conservation are given by

    (1) <c_p (s1-s0) + L_v (q1 - q0)> =  E L_v + shf + radsfc - radtop
    (2) <q1 - q0> = E - P

    Subtituting  (2) into (1) gives a more easily solved system.

    (1) <q1 - q0> = E - P
    (2) < c_p (s1 -s0) > = SHF + Lv P + radsfc - radtop

    """
    cp = 1004
    density_liquid_h2o = 1000
    latent_heat_cond = 2.51e6
    fsl_int = cp * (FSL * layer_mass).sum(-1)  # K J/K/kg * K/s kg/m2 = W/m2

    return (fsl_int - density_liquid_h2o * latent_heat_cond * prec - shf -
            radsfc + radtop)


def moisture_imbalance():
    """Same as energy imbalance but for moisture budget"""
    pass
