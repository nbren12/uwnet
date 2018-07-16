from uwnet.constraints import apply_linear_constraint, fix_negative_moisture, fix_moisture_imbalance
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


@pytest.mark.skip()
def test_precip_functional():
    lhf = torch.FloatTensor([1, 10]).unsqueeze(-1)

    z = torch.linspace(0, 16e3, 34)
    w = torch.ones(34)
    fq = (np.exp(-z/1e3) ) * 100
    fq = torch.stack([fq, fq])

    def linear(x):
        return - (w * x).sum(-1, keepdim=True) / 1000.

    a = - lhf * 86400 /  2.51e6



    fqt_modified = apply_linear_constraint(linear, a, fq, inequality=True).data
    prec_modified = precip_from_q(fqt_modified, lhf, w.data)
    prec = precip_from_q(fq.data, lhf, w.data)
    eps = 1e-7

    print("original precip", prec)
    print("modified precip", prec_modified)
    assert (prec_modified > -eps).all()


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


def test_fix_moisture_imbalance():
    n = 11
    dt = 1.0/86400  # day


    q = torch.zeros(n)
    q1 = torch.rand(n)
    layer_mass = torch.rand(n) + .5

    target_mass = 1000 # g/day/m2
    fqt = torch.rand(n) * 10
    fqt = fqt - ((fqt * layer_mass).mean())/layer_mass \
          + target_mass/layer_mass/n

    horz = (fqt * layer_mass).sum(-1)
    np.testing.assert_allclose(horz, target_mass, rtol=1e-5)

    precip = 4   # mm/day
    lhf = 400    # W/m2
    rhow = 1000  # kg/m3
    net_evap = lhf / 2.51e6 - precip / 1000 / 86400 * rhow  + \
               target_mass / 1000 / 86400


    q1_adj = fix_moisture_imbalance(q, q1, fqt, precip, lhf, dt, layer_mass)
    delta_q = (layer_mass * q1_adj / 1000).sum()
    np.testing.assert_allclose(delta_q.item(), net_evap, rtol=1e-6)
