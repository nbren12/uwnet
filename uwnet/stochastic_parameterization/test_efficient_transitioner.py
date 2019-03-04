import torch
from uwnet.tensordict import TensorDict
import numpy as np
from uwnet.stochastic_parameterization.residual_stochastic_state_model import (
    StochasticStateModel,
)
from uwnet.stochastic_parameterization.utils import get_dataset

model = StochasticStateModel(dt_seconds=15 * 60)
model.train()
state = torch.load('/Users/stewart/Desktop/state.pt')
ds = get_dataset(
    binning_method=model.binning_method,
    t_start=50,
    t_stop=75)


def get_state_for_time(time_):
    ds_filtered = ds.isel(time=time_)
    to_predict = {}
    for key_ in ds.data_vars:
        if len(ds_filtered[key_].values.shape) == 2:
            val = ds_filtered[key_].values[np.newaxis, :, :]
        else:
            val = ds_filtered[key_].values.astype(np.float64)
        to_predict[key_] = torch.from_numpy(val)
    return TensorDict(to_predict)


state = get_state_for_time(3)
efficient_probs = \
    model.eta_transitioner.get_transition_probabilities_efficient(
        model.eta, state)
true_probs = \
    model.eta_transitioner.get_transition_probabilities_true(
        model.eta, state)
