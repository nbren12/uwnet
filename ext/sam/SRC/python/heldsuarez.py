"""Held Suarez forcing module"""
import numpy as np
import logging


def call_held_suarez(state):
    log = logging.getLogger(__name__)
    log.info("Calling Held Suarez")

    # get necessary inputs from state
    lat = np.deg2rad(state['lat'])
    surface_pressure = state['pi'][0]
    top_pressure = state['pi'][-1]
    pres = (state['p'])
    pres.shape = (-1, 1, 1)
    sigma = (pres - top_pressure) / (surface_pressure - top_pressure)
    coef_sigma = coef(sigma)
    tabs = state['TABS']
    u = state['U']
    v = state['V']
    dt = state['dt']

    # temperature equation
    dtabs_y = 60.0
    dtheta_z = 10.0
    p0 = 1000.0
    kappa = 2 / 7

    t_eq = (315 - dtabs_y * np.sin(lat)**2 -
            dtheta_z * np.log(pres / p0) * np.cos(lat)**2) * (pres / p0)**kappa

    t_eq = np.maximum(200.0, t_eq)

    k_a = 1 / 40 / 86400
    k_s = 1 / 4 / 86400
    k_t = k_a + np.cos(lat)**4 * (k_s - k_a) * coef_sigma
    f_t = -k_t * (tabs - t_eq)

    # momentum equation
    k_f = 1 / 86400
    f_u = -k_f * coef_sigma * u
    f_v = -k_f * coef_sigma * v

    state.update({
        'U': u + dt * f_u,
        'V': v + dt * f_v,
        'SLI': state['SLI'] + dt * f_t
    })


def coef(sig, sig_b=.7):
    return np.maximum(0.0, (sig - sig_b) / (1 - sig_b))
