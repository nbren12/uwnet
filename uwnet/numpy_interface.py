from functools import partial

import numpy as np
from toolz import curry, pipe

import torch

from .tensordict import TensorDict


def _numpy_dict_to_torch_dict(inputs):
    return TensorDict(
        {key: torch.from_numpy(val)
         for key, val in inputs.items()})


def _torch_dict_to_numpy_dict(output):
    return {key: val.numpy() for key, val in output.items()}


def call_with_numpy_dict(self, inputs, **kwargs):
    """Call the neural network with numpy inputs"""
    with torch.no_grad():
        return pipe(inputs, _numpy_dict_to_torch_dict,
                    partial(self, **kwargs), _torch_dict_to_numpy_dict)


@curry
def NumpyWrapper(model, state):

    # Pre-process the inputs
    kwargs = {}
    for key, val in state.items():
        if isinstance(val, np.ndarray):
            kwargs[key] = val

    return call_with_numpy_dict(model, kwargs)
