import torch
from .model import rhs_hidden_from_state_dict


def test_rhs_hidden_from_state_dict():

    keys = ['ad', 'rhs.mlp.0.bias', 'rhs.mlp.2.bias', 'rhs.mlp.4.bias']
    layer_sizes = [10, 11]

    n_ad = 2
    n_out = 5

    layer_sizes_init = [n_ad] + [10, 11] + [n_out]

    state = {key: torch.zeros(n) for key, n in zip(keys, layer_sizes_init)}

    assert rhs_hidden_from_state_dict(state) == layer_sizes
