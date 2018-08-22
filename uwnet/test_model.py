import numpy as np
import torch
from uwnet.model import MLP, MOE, VariableList
from uwnet.utils import select_time, get_batch_size, stack_dicts

import pytest

sl_name = 'SLI'


def _assert_all_close(x, y):
    np.testing.assert_allclose(y.detach().numpy(), x.detach().numpy())


def _mock_batch(n, nt, nz, init=torch.rand):
    return {
        'LHF': init(n, nt),
        'SHF': init(n, nt),
        'SOLIN': init(n, nt),
        'QT': init(n, nt, nz),
        'SLI': init(n, nt, nz),
        'FQT': init(n, nt, nz),
        'FSLI': init(n, nt, nz),
        'layer_mass': torch.arange(1, nz + 1) * 1.0,
        # 'p': init(nz),
    }


def test_select_time():
    batch = _mock_batch(100, 10, 11)
    ibatch = select_time(batch, 0)

    assert ibatch[sl_name].size() == (100, 11)
    assert ibatch['SHF'].size() == (100, )


def test_get_batch_size():
    batch = _mock_batch(100, 10, 11)
    assert get_batch_size(batch) == 100


def test_stack_dicts():
    batches = _mock_batch(3, 4, 5)

    seq = [select_time(batches, i) for i in range(4)]
    out = stack_dicts(seq)

    assert out[sl_name].size() == batches[sl_name].size()
    print(out.keys())
    for key in out:
        if key != 'layer_mass':
            _assert_all_close(out[key], batches[key])


@pytest.mark.parametrize('add_forcing', [True, False])
def test_MLP_step(add_forcing):
    qt_name = 'QT'
    batch = _mock_batch(1, 1, 34)
    mlp = MLP({}, {}, time_step=.125, add_forcing=add_forcing)
    x = {}
    for key, val in batch.items():
        try:
            x[key] = val[:, 0]
        except IndexError:
            x[key] = val

    # a 0 second step should not change state
    out, _ = mlp.step(x, 0.0)
    _assert_all_close(out[qt_name], x[qt_name])

    # a 30 second step should
    with pytest.raises(AssertionError):
        out, _ = mlp.step(x, 30)
        _assert_all_close(out[qt_name], x[qt_name])


def test_mlp_forward():
    batch = _mock_batch(1, 4, 34)

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


def test_variable_input():
    nz = 5
    batch = _mock_batch(3, 4, nz)

    # rename LHF to a
    batch['a'] = batch.pop('LHF')

    mlp = MLP(
        {}, {},
        time_step=.125,
        inputs=[('a', 1), ('SLI', nz), ('QT', nz)],
        outputs=[('SLI', nz), ('QT', nz), ('SHF', 1)])

    outputs = mlp(batch)
    assert outputs['SLI'].size(-1) == nz

    for var in mlp.outputs:
        assert outputs[var.name].size(-1) == var.num


def test_to_dict():
    mlp = MLP({}, {}, time_step=.125, inputs=(('LHF', 1), ))
    a = mlp.to_dict()
    mlp1 = mlp.from_dict(a)

    assert mlp.inputs == mlp1.inputs
    assert mlp.outputs == mlp1.outputs


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_mlp_no_cheating(n):
    """Make sure that MLP only uses the initial point"""

    batch = _mock_batch(3, 10, 34)

    mlp = MLP({}, {}, time_step=.125)

    for key, val in batch.items():
        if key not in {'FSLI', 'FQT'}:
            val.requires_grad = True

    pred = mlp(batch, n=n)

    # backprop
    sl = pred['SLI']
    sl[0, -1, 0].backward()

    sl_true = batch['SLI']
    # the gradient of sl_true should only be nonzero for the first n points
    # of the data
    grad_t = torch.norm(sl_true.grad[0, :, :], dim=1)
    assert torch.sum(grad_t[n:]).item() == 0
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
