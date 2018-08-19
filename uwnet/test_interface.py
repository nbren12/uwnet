import numpy as np
import pytest
import torch

from .interface import (numpy_dict_to_torch_dict, step_with_numpy_inputs)


def test_step_with_numpy_inputs():
    def step(x, dt):
        return {'x': x['x'], 'prec': x['x'][..., 0:1]}, None

    nz = 10
    shape_3d = (nz, 3, 4)
    shape_2d = (1, 3, 4)

    dt = 0.0
    kwargs = {'x': np.ones((shape_3d))}

    out = step_with_numpy_inputs(step, kwargs, dt=None)
    out['x'].shape == shape_3d
    out['prec'].shape == shape_2d

    np.testing.assert_array_equal(out['x'], kwargs['x'])


def test_numpy_dict_to_torch_dict():
    nz, ny, nx = (10, 4, 5)
    shape_3d = (nz, ny, nx)
    shape_2d = (1, ny, nx)

    x = {'a': np.ones(shape_3d), 'b': np.ones(shape_2d), 'c': np.ones((4, ))}
    y = numpy_dict_to_torch_dict(x)

    # check sizes
    assert y['a'].size() == (ny * nx, nz)
    assert y['b'].size() == (ny * nx, 1)
    assert y['c'].size() == (4, )

    # check that outputs are torch tensors
    for key, val in y.items():
        if not torch.is_tensor(val):
            raise ValueError(key + " is not a torch tensor")
