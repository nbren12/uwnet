import os
import numpy as np

from uwnet.model import MLP
import uwnet.interface
import torch
import debug
import logging

zarr_logger = debug.ZarrLogger(os.environ.get('UWNET_ZARR_PATH', 'dbg.zarr'))

# global variables
STEP = 0

# FLAGS
DEBUG = os.environ.get('UWNET_DEBUG', '')

# open model data
OUTPUT_INTERVAL = int(os.environ.get('UWNET_OUTPUT_INTERVAL', '0'))

try:
    model_dict = torch.load(os.environ['UWNET_MODEL'])
    MODEL = MLP.from_dict(model_dict['dict'])
    MODEL.eval()
except FileNotFoundError:
    pass

# open netcdf


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

    kwargs = {}
    dt = state.pop('dt')
    for key, val in state.items():
        if isinstance(val, np.ndarray):
            kwargs[key] = val

    state['SOLIN'] = compute_insolation(state['lat'], state['day'])

    nz = MODEL.inputs.to_dict()['QT']
    kwargs = get_lower_atmosphere(kwargs, nz)

    out = uwnet.interface.step_with_numpy_inputs(MODEL.step, kwargs, dt)

    out = expand_lower_atmosphere(
        state, out, n_in=nz, n_out=state['QT'].shape[0])

    nstep = int(state['nstep'])
    output_this_step = OUTPUT_INTERVAL and (nstep - 1) % OUTPUT_INTERVAL == 0
    if DEBUG:
        save_debug({
            'args': (kwargs, dt),
            'out': out,
        }, state)

    if output_this_step:
        zarr_logger.append_all(kwargs)
        zarr_logger.append_all(out)
        zarr_logger.append('time', np.array([state['day']]))

    logger.info("Mean Precip: %f" % out['Prec'].mean())
    # store output to be read by the fortran function
    # sl and qt change names unfortunately
    # logger = get_zarr_logger()
    # for key in logger.root:
    #     logger.set_dims(key, meta.dims[key])
    state.update(out)


def call_save_debug(state):
    save_debug({'args': (state, state['dt'])}, state)
