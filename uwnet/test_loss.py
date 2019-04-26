from functools import partial

import numpy as np
import pytest
import torch
from pytest import approx
from .timestepper import Batch, TensorDict

from .loss import (compute_multiple_step_loss, r2_score,
                   weighted_mean_squared_error, weighted_r2_score,
                   multiple_step_loss_step)
from uwnet.utils import mean_over_dims, mean_other_dims
from unittest.mock import Mock

approx = partial(approx, abs=1e-6)


def _time_stepping_problem():
    n = 10
    shape = (1, n)

    def criterion(x, y):
        return torch.abs(x - y).mean()

    def model(x):
        return TensorDict({'x': torch.zeros(1, requires_grad=True).float()})

    prognostics = ['x']
    batch = TensorDict(
        {'x': torch.zeros(shape).float(), 'Fx': torch.zeros(shape).float()})
    batch = Batch(batch, prognostics)
    return batch, model, criterion, n


def test_compute_multiple_step_loss():
    batch, model, criterion, n = _time_stepping_problem()
    loss = compute_multiple_step_loss(
        criterion, model, batch, 0, n - 1, 1.0)
    assert loss.item() == pytest.approx(0.0)


# @pytest.mark.skip()
def test_multiple_step():
    batch, model, criterion, n = _time_stepping_problem()

    self = Mock()
    self.model = model
    self.time_step = 1.0
    engine = Mock()
    multiple_step_loss_step(self, engine, batch)



@pytest.mark.parametrize('x,w,dim,expected', [
    (torch.rand(10), torch.rand(10), -1, 0.0),
    (torch.rand(10, 5), torch.rand(5), -1, 0.0),
    (torch.rand(5, 10), torch.rand(5), 0, 0.0),
    (torch.rand(4, 5, 6), torch.rand(5), 1, 0.0),
])
def test_weighted_mean_squared_error(x, w, dim, expected):
    loss = weighted_mean_squared_error(x, x, w, dim)
    assert loss.dim() == 0
    assert loss.item() == pytest.approx(expected)


def test_weighted_mean_squared_error_value():
    """Test the MSE function for a non-zero value"""
    a = 2
    x = torch.rand(5)
    y = x + a
    w = torch.rand(5)
    expected = a ** 2 * w.mean()
    loss = weighted_mean_squared_error(x, y, w)
    assert loss.item() == pytest.approx(expected.item())


def test_mean_over_dims():
    a = torch.rand(10)

    # single dimension
    res = mean_over_dims(a, [0])
    np.testing.assert_allclose(res.item(), a.mean().item())

    # multiple dimensions
    a = torch.rand(10, 2)
    res = mean_over_dims(a, [0, 1])
    assert res.item() == approx(a.mean().item())

    # incomplete reduction
    a = torch.rand(3, 4, 5)
    res = mean_over_dims(a, [0, 1])
    assert res.shape == (1, 1, 5)


def test_mean_other_dims():
    a = torch.rand(3, 4, 5)
    mu = mean_other_dims(a, 1)
    assert mu.shape == (1, 4, 1)


def test_weighted_r2_score():
    a = torch.rand(10, 2)
    w = torch.tensor([1.0, 1.0])

    score = weighted_r2_score(a, a, w, dim=-1)
    assert score.item() == approx(1.0)

    score = weighted_r2_score(a, a.mean(), w, dim=-1)
    assert score.item() == approx(0.0)


def test_r2_score():
    a = torch.rand(10, 2)

    score = r2_score(a, a)
    assert score.item() == approx(1.0)

    score = r2_score(a, a.mean())
    assert score.item() == approx(0.0)
