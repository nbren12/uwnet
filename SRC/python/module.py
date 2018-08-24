import os
import tarfile
import io
import numpy as np

from uwnet.model import MLP
import uwnet.interface
import xarray as xr
import torch
from toolz import merge
import debug, meta
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
try:
    data_path = os.environ['SAM_TARGET_DATA']
except KeyError:
    mode = 'nn'
else:
    print("Loading data")
    DATASET = xr.open_zarr(data_path)
    OUTPUT_PATH = os.environ['SAM_FORCING_OUTPUT']
    mode = 'post'


def save_debug(obj, state):
    path = f"{state['case']}_{state['caseid']}_{int(state['nstep']):07d}.pkl"
    print(f"Storing debugging info to {path}")
    torch.save(obj, path)


def compute_next_state():
    """Primary entry-point to this module

    This code is invoked by the driving SAM model
    """
    if mode == 'post':
        global STEP
        update_state_from_target_data(STEP)
        STEP += 1
    else:
        state['SOLIN'] = compute_insolation(state['lat'], state['day'])
        # Store past if run for the first time
        step = MODEL.step
        out = call_neural_network(step, state)
        state.update(out)


def to_normal_units_nn_columns(ds):
    """Convert output from uwnet.columns to have proper SI units"""
    scales = {
        'FQT': 1 / 86400 / 1000,
        'FSL': 1 / 86400,
        'Q1NN': 1 / 86400 / 1000,
        'qt': 1 / 1000,
        'qtOBS': 1 / 1000,
        'Q2NN': 1 / 86400 / 1000,
    }

    for key, scale in scales.items():
        if key in ds:
            ds = ds.assign(**{key: ds[key] * scale})

    return ds


def update_state_from_target_data(step):
    """Update the state with a given step of the target dataset

    This function is used to compute the large-scale forcings from inside
    the SAM model.
    """

    logger = get_zarr_logger()

    # store the metadata if it is the first step
    if step == 0:
        attrs = debug.get_metadata(state)
        logger.root.attrs.put(attrs)
        for name in meta.constant_vars:
            logger.set(name, state[name])

    def data_array_to_numpy(x):
        dim_order = ['time', 'z', 'y', 'x']

        dims = [dim for dim in dim_order if dim in x.dims]
        return x.transpose(*dims).values

    # load the current time step
    ds = DATASET.isel(time=step).pipe(to_normal_units_nn_columns)
    print("Target time step %d of %d. Time= %.2f" % (step, len(DATASET.time),
                                                     float(ds.time)))

    # update the state
    variables_to_update = 'qt sl U V W'.split()

    for key in variables_to_update:
        state[key] = data_array_to_numpy(ds[key])

    try:
        save_debug([{
            'FQT': state['FQT'],
            'FSL': state['FSL'],
            'time': float(ds.time)
        }])
    except KeyError:
        print(
            "State does not contain the correct key...skipping until next step"
        )

    for key in logger.root:
        logger.set_dims(key, meta.dims[key])


def coef(sig, sig_b=.7):
    return np.maximum(0.0, (sig - sig_b) / (1 - sig_b))


def debug_rhs(past, state):
    return {'Q1': state['Q1'] * .9999}


def call_held_suarez(state):

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

    return {
        'U': u + dt * f_u,
        'V': v + dt * f_v,
        'SLI': state['SLI'] + dt * f_t
    }


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


def meridional_mask(n, w=4, a=1):
    """A smooth meridional mask excluding the poles"""
    y = np.r_[:n]

    z = (np.tanh((y - w) / a) + np.tanh((n - w - 1 - y) / a)) / 2
    z[z < 1e-4] = 0

    z = (y > w) * (y < n - w - 1)
    return z


def apply_meridional_mask(x, y):
    """Mix x and y according to meridional mask:

    mask * x + (1-mask) y

    See Also
    --------
    meridional_mask

    """
    weight = meridional_mask(x.shape[1], w=20)

    # expand the shape so that weight is broadcastable
    weight.shape = (-1, 1)

    return weight * x + (1 - weight) * y


def call_neural_network(state):

    logger = logging.getLogger(__name__)

    kwargs = {}
    dt = state.pop('dt')
    for key, val in state.items():
        if isinstance(val, np.ndarray):
            kwargs[key] = val

    state['SOLIN'] = compute_insolation(state['lat'], state['day'])
    out = uwnet.interface.step_with_numpy_inputs(MODEL.step, kwargs, dt)
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

    logger.info("Mean Precip: %f" %  out['Prec'].mean())
    # store output to be read by the fortran function
    # sl and qt change names unfortunately
    # logger = get_zarr_logger()
    # for key in logger.root:
    #     logger.set_dims(key, meta.dims[key])
    state.update(out)


def call_save_debug(state):
    save_debug({'args': (state, state['dt'])}, state)
