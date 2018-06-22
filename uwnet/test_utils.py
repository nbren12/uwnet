import torch
from uwnet.utils import stack_dicts


def test_stack_dicts():

    n = 10
    ins = [{'a': torch.ones(2, 1)} for i in range(n)]

    out = stack_dicts(ins)
    assert out['a'].size() == (2, n, 1)
