import numpy as np
import torch
from uwnet.model import MLP, MOE
from uwnet.utils import select_time, get_batch_size, stack_dicts

import pytest


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
        'layer_mass': torch.arange(1, nz+1) * 1.0,
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
        if key != 'layer_mass':
            _assert_all_close(out[key], batches[key])


def test_mlp_forward():
    batch = _mock_batch(3, 4, 34)

    mlp = MLP({}, {}, time_step=.125)

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


def test_variable_input():
    nz = 5
    batch = _mock_batch(3, 4, nz)

    # rename LHF to a
    batch['a'] = batch.pop('LHF')

    mlp = MLP({}, {}, time_step=.125,
              inputs=[('a', 1), ('sl', nz), ('qt', nz)],
              outputs=[('sl', nz), ('qt', nz), ('SHF', 1)])

    outputs = mlp(batch)
    assert outputs['sl'].size(-1) == nz

    for key in ['sl', 'qt', 'SHF']:
        assert outputs[key].size(-1) == mlp.output_fields[key]

def test_to_dict():
    mlp = MLP({}, {}, time_step=.125, inputs=(('LHF', 1),))
    a = mlp.to_dict()
    mlp1 = mlp.from_dict(a)

    assert mlp.input_fields == mlp1.input_fields
    assert mlp.output_fields == mlp1.output_fields

@pytest.mark.parametrize("n", [1,2,3,4])
def test_mlp_no_cheating(n):
    """Make sure that MLP only uses the initial point"""

    batch = _mock_batch(3, 10, 34)

    mlp = MLP({}, {}, time_step=.125)


    for key, val in batch.items():
        val.requires_grad = True

    pred = mlp(batch, n=n)

    # backprop
    sl = pred['sl']
    sl[0, -1, 0].backward()

    sl_true = batch['sl']
    # the gradient of sl_true should only be nonzero for the first n points
    # of the data
    grad_t  = torch.norm(sl_true.grad[0, :, :], dim=1)
    assert torch.sum(grad_t[n:]).item() == 0
    assert torch.sum(grad_t[0:n]).item() > 0
