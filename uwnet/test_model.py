import numpy as np
import torch
from uwnet.model import MLP, MOE, VariableList
from uwnet.utils import stack_dicts

import pytest
from pytest import approx

sl_name = 'SLI'


def _assert_all_close(x, y):
    np.testing.assert_allclose(y.detach().numpy(), x.detach().numpy())


def _mock_batch(t, z, y, x, init=torch.rand):
    return {
        'LHF': init(t,y,x),
        'SHF': init(t,y,x),
        'SOLIN': init(t,y,x),
        'QT': init(t, z, y, x),
        'SLI': init(t, z, y, x),
        'FQT': init(t, z, y, x),
        'FSLI': init(t, z, y, x),
        'layer_mass': torch.arange(1, z + 1).float(),
        # 'p': init(nz),
    }


@pytest.mark.parametrize('add_forcing', [True, False])
def test_MLP_step(add_forcing):
    qt_name = 'QT'
    batch = _mock_batch(100, 34, 4, 5)
    mlp = MLP({}, {}, time_step=.125, add_forcing=add_forcing)
    x = {}
    for key, val in batch.items():
        try:
            x[key] = val[ 0]
        except IndexError:
            x[key] = val

    # a 0 second step should not change state
    out = mlp.step(x, 0.0)
    _assert_all_close(out[qt_name], x[qt_name])

    # a 30 second step should
    with pytest.raises(AssertionError):
        out = mlp.step(x, 30)
        _assert_all_close(out[qt_name], x[qt_name])


def test_mlp_forward():
    batch = _mock_batch(1, 34, 10, 10)

    mlp = MLP({}, {}, time_step=.125)

    pred = mlp(batch, n=1)

    assert pred['SLI'].size() == batch['QT'].size()


def test_moe():

    m = 10
    n = 5
    n_exp = 3

    x = torch.rand(100, m)

    rhs = MOE(m, n, n_exp)
    out = rhs(x)
    return out.size() == (100, n)


def test_to_dict():
    mlp = MLP({}, {}, time_step=.125, inputs=(('LHF', 1), ))
    a = mlp.to_dict()
    mlp1 = mlp.from_dict(a)

    assert mlp.inputs == mlp1.inputs
    assert mlp.outputs == mlp1.outputs


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_mlp_no_cheating(n):
    """Make sure that MLP only uses the initial point"""

    batch = _mock_batch(10, 34, 10,10)

    mlp = MLP({}, {}, time_step=.125)

    # mock out the step method
    def mock_step(x, dt, *args):
        return {key: val + 1.0 for key, val in x.items()}
    mlp.step = mock_step

    for key, val in batch.items():
        if key not in {'FSLI', 'FQT'}:
            val.requires_grad = True

    pred = mlp(batch, n=n)

    # backprop
    sl = pred['SLI']
    sl[-1, 0, 0, 0].backward()

    sl_true = batch['SLI']
    # the gradient of sl_true should only be nonzero for the first n points
    # of the dat
    grad_t = (sl_true.grad**2).mean(-1).mean(-1).mean(-1)
    assert torch.sum(grad_t[n:]).item() == approx(0)
    assert torch.sum(grad_t[0:n]).item() > 0


def test_VariableList():

    inputs = [('LHF', 1), ('SHF', 1), ('SOLIN', 1), ('qt', 34), (sl_name, 34)]
    vl = VariableList.from_tuples(inputs)
    assert vl.num == 3 + 2 * 34

    inputs = [('SHF', 1, True)]
    vl = VariableList.from_tuples(inputs)

    # see if indexing works
    v = vl[0]
    assert v.name == 'SHF'
    assert v.positive


def test_VariableList_stack_unstack():
    # check stacking and unstacking
    inputs = [('a', 1), ('b', 10)]
    data = {'a': torch.rand(11, 1), 'b': torch.rand(11, 10)}
    vl = VariableList.from_tuples(inputs)

    b = vl.stack(data)
    assert b.shape == (11, 11)

    data_orig = vl.unstack(b)
    for key in data_orig:
        np.testing.assert_equal(data_orig[key].numpy(), data[key].numpy())
