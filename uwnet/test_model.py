import numpy as np
import pytest
import torch
from pytest import approx
from torch.nn import Module

import xarray as xr
from uwnet.model import (MOE, ApparentSource, ForcedStepper, VariableList,
                         call_with_xr)

sl_name = 'SLI'


def _assert_all_close(x, y):
    np.testing.assert_allclose(y.detach().numpy(), x.detach().numpy())


def _mock_batch(t, z, y, x, init=torch.rand):
    return {
        'LHF': init(t, y, x),
        'SHF': init(t, y, x),
        'SOLIN': init(t, y, x),
        'QT': init(t, z, y, x),
        'SLI': init(t, z, y, x),
        'FQT': init(t, z, y, x),
        'FSLI': init(t, z, y, x),
        'layer_mass': torch.arange(1, z + 1).float(),
        # 'p': init(nz),
    }


def test_moe():

    m = 10
    n = 5
    n_exp = 3

    x = torch.rand(100, m)

    rhs = MOE(m, n, n_exp)
    out = rhs(x)
    return out.size() == (100, n)


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


def _stepper(z):
    return ForcedStepper(
        forcing=('QT', 'SLI'),
        auxiliary=(('LHF', 1), ('SHF', 1), ('SOLIN', 1)),
        prognostic=(('QT', z), ('SLI', z)),
        diagnostic=(),
        mean={},
        scale={},
        time_step=3600.0)


def test_ForcedStepper():
    z = 34
    batch = _mock_batch(10, z, 10, 10)
    stepper = _stepper(z)
    out = stepper(batch, n=4)
    y = out['QT'].mean()
    y.backward()

    # test to_dict
    ForcedStepper.from_dict(stepper.to_dict())


def test_stepper_no_cheating():
    """Make sure that stepper only uses the initial point"""

    n = 10
    z = 34
    batch = _mock_batch(10, z, 10, 10)
    stepper = _stepper(z)

    # mock out the forward method
    class MockStep(Module):
        def forward(self, x):
            return {key: val + 1.0 for key, val in x.items()}

    stepper.rhs = MockStep()

    for key, val in batch.items():
        if key not in {'FSLI', 'FQT'}:
            val.requires_grad = True

    pred = stepper(batch)

    # backprop
    sl = pred['SLI']
    sl[-1, 0, 0, 0].backward()

    sl_true = batch['SLI']
    # the gradient of sl_true should only be nonzero for the first n points
    # of the dat
    grad_t = (sl_true.grad**2).mean(-1).mean(-1).mean(-1)
    assert torch.sum(grad_t[n:]).item() == approx(0)
    assert torch.sum(grad_t[0:n]).item() > 0


def test_ApparentSource():
    mod = ApparentSource(
        mean={}, scale={}, inputs=(('a', 34), ), outputs=(('a', 34), ))

    batch = {'a': torch.ones(128, 34, 1, 1)}
    y = mod(batch)
    a = y['a']
    assert a.shape == batch['a'].shape


@pytest.mark.skip()
@pytest.mark.parametrize('drop_time', [0, 1, 2])
def test_call_with_xr(drop_time):
    def model(d):
        return {key: val[drop_time:] for key, val in d.items()}

    dims = ['time', 'z', 'y', 'x']
    shape = [10, 11, 12, 13]
    coords = {dim: np.arange(n) for dim, n in zip(dims, shape)}

    ds = xr.Dataset({'a': (dims, np.random.rand(*shape))}, coords=coords)
    out = call_with_xr(model, ds, drop_times=drop_time)
    np.testing.assert_array_equal(ds.a.values[drop_time:], out.a.values)
