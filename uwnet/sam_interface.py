"""Interface for calling neural network models from SAM

Configuration
-------------

This interface routine is configured using environmental variables. Currently
these are

- UWNET_MODEL : The path to the model in a pickle file


Nudging
~~~~~~~

UWNET_NUDGE_TIME_SCALE: nudging time-scale
NGAQUA_PATH : path to target ngaqua data

"""
# import debug
import logging
import os

import numpy as np
from toolz import curry, valmap

import torch
from uwnet.numpy_interface import NumpyWrapper
from uwnet.sam_ngaqua import get_ngaqua_nudger


def get_configuration_from_environment():
    models = []
    try:
        model = {"type": "neural_network", "path": os.environ['UWNET_MODEL']}
        models.append(model)
    except KeyError:
        pass

    try:
        models.append({
            'type':
            'nudging',
            'time_scale':
            float(os.environ['UWNET_NUDGE_TIME_SCALE']),
            'ngaqua':
            os.environ['NGAQUA_PATH']
        })
    except KeyError:
        pass

    return {'models': models}


def rename_keys(rename_table, d):
    return {rename_table.get(key, key): d[key] for key in d}


@curry
def CFVariableNameAdapter(model, d, label=''):
    """Wrapper for translating input/output variable names in the neural network
    model to CF-compliant ones"""

    table = [
        ("liquid_ice_static_energy", "SLI"),
        ("x_wind", "U"),
        ("y_wind", "V"),
        ("upward_air_velocity", "W"),
        ("total_water_mixing_ratio", "QT"),
        ("air_temperature", "TABS"),
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("sea_surface_temperature", "SST"),
        ("surface_air_pressure", "p0"),
        ("toa_incoming_shortwave_flux", "SOLIN"),
        ("surface_upward_sensible_heat_flux", "SHF"),
        ("surface_upward_latent_heat_flux", "LHF"),
    ]

    input_keys = dict(table)
    output_keys = dict((y, x) for x, y in table)

    # Translate from CF input names to the QT, SLI, names
    d_with_old_names = rename_keys(input_keys, d)
    out = model(d_with_old_names)

    # translate output keys to CF
    out = rename_keys(output_keys, out)
    # add tendency_of to these keys
    tendency_names = {
        key: 'tendency_of_' + key + '_due_to_' + label
        for key in out
    }

    return rename_keys(tendency_names, out)


def get_model(config):
    type = config['type']
    if type == 'neural_network':
        model = torch.load(config['path'])
        model.eval()
        return CFVariableNameAdapter(
            NumpyWrapper(model), label='neural_network')
    elif type == 'nudging':
        return CFVariableNameAdapter(
            get_ngaqua_nudger(config), label='nudging')
    else:
        raise NotImplementedError(f"Model type {type} not implemented")


CONFIG = get_configuration_from_environment()
MODELS = [get_model(model) for model in CONFIG['models']]


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
    return scon * eccf * mu


def sum_up_tendencies(d):
    import re
    output = {}

    pattern = re.compile('tendency_of_(.*)_due_to')

    for key in d:
        m = pattern.search(key)
        if m:
            variable_name = 'tendency_of_' + m.group(1)
        else:
            variable_name = key

        seq = output.setdefault(variable_name, [])
        seq.append(d[key])

    return valmap(sum,  output)


def call_neural_network(state):

    logger = logging.getLogger(__name__)

    # compute insolation. it is difficult to configure SAM to compute this
    # without including radiation
    state['SOLIN'] = compute_insolation(state['latitude'], state['day'])

    # Compute the output of all the models
    all_outputs = {}
    for model in MODELS:
        logger.info(f"Calling {model}")
        out = model(state)
        all_outputs.update(out)

    totaled_output = sum_up_tendencies(all_outputs)

    # update the state
    state.update(totaled_output)
    state.update(all_outputs)


def call_save_state(state):
    """This simple function can be called in SAM using the following namelist
    entries:

    module_name = <this modules name>,
    function_name = 'call_save_state'

    """
    step = int(state.get('nstep', 0))
    caseid = state['caseid']
    case = state['case']
    name = f"OUT_3D/{case}_{caseid}_{step:010d}.pt"
    torch.save(state, name)
