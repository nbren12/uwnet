import numpy as np
import pytest
import torch
from toolz import curry
from .testing import assert_tensors_allclose, mock_data
from .normalization import Scaler
from uwnet.tensordict import TensorDict

approx = curry(pytest.approx, abs=1e-6)


def test_scaler():
    x = torch.rand(10)
    mean = x.mean()
    scale = x.std()

    scaler = Scaler({'x': mean}, {'x': scale})
    y = scaler({'x': x})
    scaled = y['x']
    assert scaled.mean().item() == approx(0.0)
    assert scaled.std().item() == approx(1.0)


def test_scaler_fit_xarray():
    name = 'a'
    ds = mock_data(init=np.random.random).to_dataset(name=name)
    scaler = Scaler().fit_xarray(ds)

    expected = torch.from_numpy(ds.groupby('z').mean(...)[name].values)
    assert_tensors_allclose(scaler.mean[name], expected)


def test_scaler_fit_generator():
    shape = (10, 3, 4, 1, 1)
    name = 'a'
    a = torch.rand(shape)

    def generator():
        for arr in a.split(2, dim=0):
            yield TensorDict({name: a})

    scaler = Scaler().fit_generator(generator())

    expected_mean = a.mean(0).mean(0).squeeze()
    assert_tensors_allclose(scaler.mean[name], expected_mean)
