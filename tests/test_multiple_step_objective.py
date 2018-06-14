import numpy as np
import torch
from lib.torch.model import _fix_moisture

import pytest


def test__fix_moisture():
    z = torch.linspace(0, 16e3, 34)
    w = torch.ones_like(z)

    q = torch.exp(-z/4e3) - .2

    q_new = _fix_moisture(q, w)

    np.testing.assert_allclose((q *w).sum(), (q_new * w).sum())
    assert (q_new > 0).all()
