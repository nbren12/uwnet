import numpy as np
import torch
from uwnet.model import MLP, MOE
from uwnet.utils import select_time, get_batch_size, stack_dicts


def _assert_all_close(x, y):
    np.testing.assert_allclose(y.numpy(), x.numpy())


def _mock_batch(n, nt, nz, init=torch.rand):
    return {
        'LHF': init(n, nt),
        'SHF': init(n, nt),
        'SOLIN': init(n, nt),
        'qt': init(n, nt, nz),
        'sl': init(n, nt, nz),
        'FQT': init(n, nt, nz),
        'FSL': init(n, nt, nz),
        # 'p': init(nz),
    }


def test_select_time():
    batch = _mock_batch(100, 10, 11)
    ibatch = select_time(batch, 0)

    assert ibatch['sl'].size() == (100, 11)
    assert ibatch['SHF'].size() == (100, )


def test_get_batch_size():
    batch = _mock_batch(100, 10, 11)
    assert get_batch_size(batch) == 100


def test_stack_dicts():
    batches = _mock_batch(3, 4, 5)

    seq = [select_time(batches, i) for i in range(4)]
    out = stack_dicts(seq)

    assert out['sl'].size() == batches['sl'].size()
    print(out.keys())
    for key in out:
        print(key)
        _assert_all_close(out[key], batches[key])


def test_mlp_forward():
    batch = _mock_batch(3, 4, 34)

    mlp = MLP({}, {})

    pred = mlp(batch, n=1)

    assert pred['sl'].size() == batch['qt'].size()


def test_moe():

    m = 10
    n = 5
    n_exp = 3

    x = torch.rand(100, m)

    rhs = MOE(m, n, n_exp)
    out = rhs(x)
    return out.size() == (100, n)
