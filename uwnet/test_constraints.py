from uwnet.constraints import apply_linear_constraint
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

