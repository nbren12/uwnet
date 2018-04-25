import numpy as np
import torch
from lib.torch.model import (
    enforce_precip_sl, precip_from_s, enforce_precip_qt, precip_from_q,
    padded_deriv, _fix_moisture)

import pytest


@pytest.mark.skip()
def test_enforce_precip_sl():
    n, m = (100, 34)
    fsl = torch.rand(n, m)
    qrad = torch.rand(n, m)
    shf = torch.rand(n, 1)
    precip = torch.ones(n, 1)
    w = torch.ones(m)

    fsl_adj = enforce_precip_sl(fsl, qrad, shf, precip, w)
    prec_calc = precip_from_s(fsl_adj, qrad, shf, w)

    np.testing.assert_allclose(prec_calc.numpy(), precip.numpy(), atol=1e-6)


@pytest.mark.skip()
def test_enforce_precip_qt():
    n, m = (100, 34)
    fqt = torch.rand(n, m)
    lhf = torch.rand(n, 1)
    precip = torch.ones(n, 1)
    w = torch.ones(m)

    fqt_adj = enforce_precip_qt(fqt, lhf, w)
    prec_calc = precip_from_q(fqt_adj, lhf, w)

    np.testing.assert_allclose(prec_calc.numpy(), precip.numpy(), atol=1e-7)


def test_vertical_avection():
    z = torch.arange(0, 10)
    ones = (z * 3).view(1, -1)
    df = padded_deriv(ones, z)
    np.testing.assert_allclose(df.numpy(), 3)


def test__fix_moisture():
    z = torch.linspace(0, 16e3, 34)
    w = torch.ones_like(z)

    q = torch.exp(-z/4e3) - .2

    q_new = _fix_moisture(q, w)

    np.testing.assert_allclose((q *w).sum(), (q_new * w).sum())
    assert (q_new > 0).all()
