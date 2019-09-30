import logging

import joblib
import numpy as np
import json

from uwnet.thermo import compute_insolation

def initialize(state):
    pass


def finalize(state):
    pass


def get_configuration():
    try:
        with open("python_config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {'models': []}



def get_model(config):
    type = config['type']
    if type=='sklearn_generic':
        model = joblib.load(config['path'])
        return model
    else:
        raise NotImplementedError(f"Model type {type} not implemented")


def state_format_to_sklearn_format(
        state,
        key_to_extract
):
    """
    takes the current state dict and returns single feature z profile np array input for sklearn_generic model
    """
    array_3d = state[key_to_extract]
    zdim, ydim, xdim = np.shape(array_3d)
    sklearn_format_matrix = array_3d \
        .transpose(2, 1, 0) \
        .reshape(xdim*ydim, zdim)
    return sklearn_format_matrix

def sklearn_format_to_state_format(
        model_output_array,
        state_arr_dims=(34, 64, 128)
):
    """
    assume the flattening order for input is y then x
    """
    zdim, ydim, xdim = state_arr_dims
    z_flattened_model = np.array(model_output_array).flatten()
    state_value = np.reshape(z_flattened_model , (zdim, ydim, xdim), 'F')
    return state_value


CONFIG = get_configuration()
MODELS =[get_model(model) for model in CONFIG['models']]


def call_sklearn_model(state):
    logger = logging.getLogger(__name__)
    # compute insolation. it is difficult to configure SAM to compute this
    # without including radiation
    state['SOLIN'] = compute_insolation(state['latitude'], state['day'])

    # get feature matrix out of state
    sl_sklearn_format = state_format_to_sklearn_format(
        state,
        key_to_extract='liquid_ice_static_energy'
    )
    qt_sklearn_format = state_format_to_sklearn_format(
        state,
        key_to_extract='total_water_mixing_ratio'
    )
    solin_sklearn_format = state_format_to_sklearn_format(
        state,
        key_to_extract='SOLIN'
    )
    sst_sklearn_format = state_format_to_sklearn_format(
        state,
        key_to_extract='sea_surface_temperature'
    )
    feature_matrix = np.hstack([
        sl_sklearn_format,
        qt_sklearn_format,
        solin_sklearn_format,
        sst_sklearn_format
    ])

    # get model output
    sklearn_model = MODELS[0]
    predicted_q1_q2 = sklearn_model.predict(feature_matrix)
    q1_state = sklearn_format_to_state_format(
        predicted_q1_q2[:, :34],
        state_arr_dims=(34, 64, 128)
    )
    q2_state = sklearn_format_to_state_format(
        predicted_q1_q2[:, 34:],
        state_arr_dims=(34, 64, 128)
    )
    # update keys 'tendency_of_...'
    state['tendency_of_liquid_ice_static_energy'] = q1_state
    state['tendency_of_total_water_mixing_ratio'] = q2_state