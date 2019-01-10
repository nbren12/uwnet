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


def get_models():
    """Load the specified torch models

    The environmental variable UWNET_MODEL should point to the model for Q1 and
    Q2. If present MOM_MODEL should point to the momentum source model.
    """
    models = []

    # Load Q1/Q2 model
    model_path = os.environ['UWNET_MODEL']
    model = torch.load(model_path)

    if isinstance(model, nn.Module):
        model.eval()
        model = CFVariableNameAdapter(NumpyWrapper(model))

    models.append(model)
    return models


# zarr_logger = debug.ZarrLogger(os.environ.get('UWNET_ZARR_PATH', 'dbg.zarr'))

# global variables
STEP = 0
try:
    MODELS = get_models()
except KeyError:
    print(
        "Model not loaded...need set the UWNET_MODEL environmental variable.")

# FLAGS
DEBUG = os.environ.get('UWNET_DEBUG', '')

# open model data
OUTPUT_INTERVAL = int(os.environ.get('UWNET_OUTPUT_INTERVAL', '0'))


def save_debug(obj, state):
    path = f"{state['case']}_{state['caseid']}_{int(state['nstep']):07d}.pkl"
    print(f"Storing debugging info to {path}")
    torch.save(obj, path)


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
    dt = state.pop('dt')
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

    # Debugging info below here
    # -------------------------
    nstep = int(state['nstep'])
    output_this_step = OUTPUT_INTERVAL and (nstep - 1) % OUTPUT_INTERVAL == 0
    if DEBUG:
        save_debug({
            'args': (kwargs, dt),
            'out': merged_outputs,
        }, state)

    try:
        logger.info("Mean Precip: %f" % out['Prec'].mean())
    except KeyError:
        pass

    if output_this_step:
        zarr_logger.append_all(kwargs)
        zarr_logger.append_all(merged_outputs)
        zarr_logger.append('time', np.array([state['day']]))

    # store output to be read by the fortran function
    # sl and qt change names unfortunately
    # logger = get_zarr_logger()
    # for key in logger.root:
    #     logger.set_dims(key, meta.dims[key])


def call_save_state(state):
    """This simple function can be called in SAM using the following namelist
    entries:

    module_name = <this modules name>,
    function_name = 'call_save_state'

    """
    torch.save(state, "state.pt")
