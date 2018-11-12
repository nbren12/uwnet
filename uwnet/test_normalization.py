import torch
from .normalization import Scaler
import pytest
from toolz import curry

approx = curry(pytest.approx, abs=1e-6)


def test_Scaler():

    x = torch.rand(10)
    mean = x.mean()
    scale = x.std()

    scaler = Scaler({'x': mean}, {'x': scale})
    y = scaler({'x': x})
    scaled = y['x']
    assert scaled.mean().item() == approx(0.0)
    assert scaled.std().item() == approx(1.0)
