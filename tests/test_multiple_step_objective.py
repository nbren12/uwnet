import numpy as np
import torch
from lib.models.torch.multiple_step_objective import (
    enforce_precip_sl, precip_from_s, enforce_precip_qt, precip_from_q)


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


def test_enforce_precip_qt():
    n, m = (100, 34)
    fqt = torch.rand(n, m)
    lhf = torch.rand(n, 1)
    precip = torch.ones(n, 1)
    w = torch.ones(m)

    fqt_adj = enforce_precip_qt(fqt, lhf, precip, w)
    prec_calc = precip_from_q(fqt_adj, lhf, w)

    np.testing.assert_allclose(prec_calc.numpy(), precip.numpy(), atol=1e-7)
