from uwnet.constraints import (apply_linear_constraint, fix_negative_moisture,
                               expected_moisture, enforce_expected_integral,
                               expected_temperature)
import numpy as np
import torch

import pytest


def test_apply_linear_constraint():
    def lin(x):
        """sum(x) >= 0"""
        return x.sum(-1, keepdim=True)

    x = torch.ones(1, 10)

    # test equality constraint
    y = apply_linear_constraint(lin, 0, x).data.numpy()
    np.testing.assert_allclose(y, 0.0)

    # test inequality constraint
    y = apply_linear_constraint(lin, 0, x, inequality=True)
    np.testing.assert_almost_equal(y.data.numpy(), x.data.numpy())

    # test inequality constraint
    y = apply_linear_constraint(lin, 0, -x, inequality=True)
    np.testing.assert_allclose(y.data.numpy(), 0.0)

    # test shape
    assert y.size() == x.size()


def test_fix_negative_moisture():
    q = torch.arange(-4, 20) + .1
    layer_mass = torch.rand(len(q)) + .1

    q_new = fix_negative_moisture(q, layer_mass)

    # No negative humidiy
    assert (q_new >= 0).all()

    # water is conserved
    q1 = (q_new * layer_mass).sum(-1)
    q0 = (q * layer_mass).sum(-1)
    np.testing.assert_allclose(q1.numpy(), q0.numpy())


def test_fix_expected_moisture():
    n = 11
    q = torch.rand(n)
    q1 = torch.rand(n)

    layer_mass = torch.rand(n) + .5

    pw0, pw = expected_moisture(q, 0, 0, 0, 0, layer_mass)
    actual = (layer_mass * q).sum()
    np.testing.assert_allclose(pw.item(), actual.item())

    latent_heat = 2.51e6
    evap = 5.0  # kg/m^2/day
    lhf = evap / 86400 * latent_heat
    pw0, pw = expected_moisture(q, 0, 0, lhf, 1.0, layer_mass)
    np.testing.assert_allclose((pw-pw0).item()/1000, evap)


def test_enforce_expected_integral():
    expected = 2.0

    x = torch.rand(100)
    w = torch.rand(100) + .5
    x = enforce_expected_integral(x, expected, w)
    assert (x * w).sum().item() == pytest.approx(expected)


def test_expected_temperature():

    h = 1.0
    temp = torch.tensor(300.0)
    mass = torch.tensor(1.0)

    delta_temp = torch.tensor(1.0)

    next_temp = temp + delta_temp
    _, ans = expected_temperature(
        temp,
        delta_temp,
        prec=0,
        shf=0,
        radtoa=0,
        radsfc=0,
        layer_mass=mass,
        h=1.0)
    assert pytest.approx(ans.item()) == next_temp.item()

    # SHF
    shf = 100  # W/m2
    next_temp = temp + shf / 1004 * h * 86400
    _, ans = expected_temperature(
        temp,
        0,
        prec=0,
        shf=shf,
        radtoa=0,
        radsfc=0,
        layer_mass=mass,
        h=1.0)
    assert pytest.approx(ans.item()) == next_temp.item()

    # RADTOA
    _, ans = expected_temperature(
        temp,
        0,
        prec=0,
        shf=0,
        radtoa=-shf,
        radsfc=0,
        layer_mass=mass,
        h=1.0)
    assert pytest.approx(ans.item()) == next_temp.item()

    # RADSFC
    _, ans = expected_temperature(
        temp,
        0,
        prec=0,
        shf=0,
        radtoa=0,
        radsfc=shf,
        layer_mass=mass,
        h=1.0)
    assert pytest.approx(ans.item()) == next_temp.item()
