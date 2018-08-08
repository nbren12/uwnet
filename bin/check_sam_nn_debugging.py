#!/usr/bin/env python
import sys

import torch
import xarray as xr

from uwnet.interface import step_model
from uwnet.model import MLP

def lhf_to_evap(lhf):
    return lhf * 86400 / 2.51e6


def precipitable_water(qv, layer_mass, dim='p'):
    return (qv * layer_mass).sum(dim)


# water budget stuff
def water_budget(ds, dim='p'):
    """Compute water budget if Q2NN is present"""
    q2_int = (ds.Q2NN * ds.layer_mass).sum(dim) * 86400
    q2_ib = q2_int
    prec = ds.Prec
    evap = lhf_to_evap(ds.LHF)

    return xr.Dataset(
        dict(Prec=prec, evap=evap, Q2=q2_ib, imbalance=q2_ib - (evap - prec)))


def vertical_integral(w, x):
    return (w.reshape((-1, 1, 1)) * x).sum(axis=-3)


def print_water_budget(b):
    for key, val in b.items():
        print(key, val, "mm/day")
    print("P-E", b['Prec'] - b['Evap'], "mm/day")


def check_water_budget(state, out):
    w = state['layer_mass']
    pw0 = vertical_integral(w, state['qt']).mean()
    pw1 = vertical_integral(w, out['qt']).mean()
    dt = state['dt']

    water_budget = {
        "pw0 (mm)": pw0,
        "pw1 (mm)": pw1,
        'dPW': (pw1 - pw0) / dt * 86400,
        "Q2": vertical_integral(w, out['Q2NN']).mean() * 86400,
        "FQT": vertical_integral(w, state['FQT']).mean() * 86400,
        "Prec": out['Prec'].mean(),
        "Evap": out['LHF'].mean() / 2.51e6 * 86400,
    }

    print_water_budget(water_budget)


def check_vert_vel(state):
    for key in state:
        w = state[key]
        try:
            print(f"{key} range: {w.min():.2e}--{w.max():.2e}\t mean+sig: {w.mean():.2e} +- {w.std():.2e}")
        except AttributeError:
            pass


def load_model_from_path(path):
    return MLP.from_dict(torch.load(path)['dict'])


dbg = torch.load(sys.argv[1])
model = load_model_from_path(sys.argv[2])

state, out = dbg['kwargs'], dbg['out']
out = step_model(model.step, **state)

check_water_budget(dbg['kwargs'], out)
check_vert_vel(dbg['kwargs'])
