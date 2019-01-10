"""Interface for calling neural network models from SAM

Configuration
-------------

This interface routine is configured using environmental variables. Currently
these are

- UWNET_MODEL : The path to the model in a pickle file

"""
# import debug
import logging
import os

import numpy as np
from toolz import curry

import torch
from torch import nn
from uwnet.numpy_interface import NumpyWrapper


def get_configuration_from_environment():
    return {
        'models': [os.environ['UWNET_MODEL']]
    }


def rename_keys(rename_table, d):
    return {rename_table.get(key, key): d[key] for key in d}


@curry
def CFVariableNameAdapter(model, d):
    """Wrapper for translating input/output variable names in the neural network
    model to CF-compliant ones"""

    table = [
        ("liquid_ice_static_energy", "SLI"),
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
        key: 'tendency_of_' + key + '_due_to_neural_network'
        for key in out
    }

    return rename_keys(tendency_names, out)


def get_models(config):
    """Load the specified torch models

    The environmental variable UWNET_MODEL should point to the model for Q1 and
    Q2. If present MOM_MODEL should point to the momentum source model.
    """
    models = []

    for model_path in config['models']:
        model = torch.load(model_path)

        if isinstance(model, nn.Module):
            model.eval()
            model = CFVariableNameAdapter(NumpyWrapper(model))

    models.append(model)
    return models


CONFIG = get_configuration_from_environment()
MODELS = get_models(CONFIG)


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
    eccf : float
        eccentricity factor. Ratio of orbital radius at perihelion and
        aphelion. Default 1.0.

    """
    time_of_day = day % 1.0

    # cos zenith angle
    mu = -np.cos(2 * np.pi * time_of_day) * np.cos(np.pi * lat / 180)
    return scon * eccf * mu


def call_neural_network(state):

    logger = logging.getLogger(__name__)

    # Pre-process the inputs
    # ----------------------
    kwargs = {}
    for key, val in state.items():
        if isinstance(val, np.ndarray):
            kwargs[key] = val

    kwargs['SOLIN'] = compute_insolation(state['latitude'], state['day'])

    # Compute the output of all the models
    # ------------------------------------
    for model in MODELS:
        logger.info(f"Calling NN")
        out = model(kwargs)

    # update the state
    state.update(out)


def call_save_state(state):
    """This simple function can be called in SAM using the following namelist
    entries:

    module_name = <this modules name>,
    function_name = 'call_save_state'

    """
    torch.save(state, "state.pt")
