import os
import numpy as np

from uwnet.model import MLP
import torch
import debug
import logging
from toolz import valmap


def get_models():
    """Load the specified torch models

    The environmental variable UWNET_MODEL should point to the model for Q1 and
    Q2. If present MOM_MODEL should point to the momentum source model.
    """
    models = []

    # Load Q1/Q2 model
    model_dict = torch.load(os.environ['UWNET_MODEL'])
    MODEL = MLP.from_dict(model_dict['dict'])
    MODEL.eval()
    models.append(MODEL)

    # Load Q3 model
    try:
        model_dict = torch.load(os.environ['UWNET_MOMENTUM_MODEL'])
        MOM_MODEL = MLP.from_dict(model_dict['dict'])
        logging.info("Loaded momentum source model")
        MOM_MODEL.eval()
        models.append(MOM_MODEL)
    except:
        logging.info("Momentum source model not loaded")
        pass

    return models


zarr_logger = debug.ZarrLogger(os.environ.get('UWNET_ZARR_PATH', 'dbg.zarr'))

# global variables
STEP = 0
MODELS = get_models()

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


def get_lower_atmosphere(kwargs, nz):
    # limit vertical variables to this number of height poitns
    fields_3d = ['SLI', 'QT', 'TABS', 'W', 'FQT', 'FSLI', 'U', 'V']
    fields_z = ['layer_mass', 'p', 'pi']
    out = {}
    for field in kwargs:
        if field in fields_3d + fields_z:
            out[field] = kwargs[field][:nz]
        else:
            out[field] = kwargs[field]

    return out


def expand_lower_atmosphere(sam_state, nn_output, n_in, n_out):
    out = {}
    for key in nn_output:
        if nn_output[key].shape[0] == n_in:
            if key in sam_state:
                upper_atmos = sam_state[key][n_in:]
                lower_atmos = nn_output[key]
                out[key] = np.concatenate([lower_atmos, upper_atmos], axis=0)
            else:
                out[key] = np.pad(
                    lower_atmos, [(0, n_out - n_in), (0, 0), (0, 0)],
                    mode='constant')

            assert out[key].shape[0] == n_out
        else:
            out[key] = nn_output[key]

    return out


def call_neural_network(state):

    logger = logging.getLogger(__name__)

    # Pre-process the inputs
    # ----------------------
    kwargs = {}
    dt = state.pop('dt')
    for key, val in state.items():
        if isinstance(val, np.ndarray):
            kwargs[key] = val

    state['SOLIN'] = compute_insolation(state['lat'], state['day'])

    # Compute the output of all the models
    # ------------------------------------
    merged_outputs = {}
    for model in MODELS:
        logger.info(f"Calling {model}")
        nz = model.inputs.to_dict()['QT']
        lower_atmos_kwargs = get_lower_atmosphere(kwargs, nz)

        # add a singleton dimension and convert to float32
        lower_atmos_kwargs = {key: val[np.newaxis].astype(np.float32) for key,
                              val in lower_atmos_kwargs.items()}
        # call the neural network
        out = model.call_with_numpy_dict(lower_atmos_kwargs, n=1, dt=float(dt))
        # remove the singleton first dimension
        out = valmap(np.squeeze, out)
        out = expand_lower_atmosphere(
            state, out, n_in=nz, n_out=state['QT'].shape[0])

        merged_outputs.update(out)

    # update the state
    state.update(merged_outputs)

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


def call_save_debug(state):
    """This simple function can be called in SAM using the following namelist
    entries:

    module_name = <this modules name>,
    function_name = 'call_save_debug'

    """
    save_debug({'args': (state, state['dt'])}, state)
