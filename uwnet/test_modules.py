import torch
from torch import nn
from .modules import LinearDictIn, LinearDictOut, LinearFixed, MapByKey
from .tensordict import TensorDict
import pytest
from .testing import assert_tensors_allclose


def test_LinearDictIn():
    inputs = [('A', 10), ('B', 5)]
    n_out = 4

    lin = LinearDictIn(inputs, n_out)

    data = {'A': torch.zeros(1, 10), 'B': torch.zeros(1, 5)}
    out = lin(data)
    assert out.size(-1) == n_out
    out[0, 0].backward()


def test_LinearDictOut():
    outputs = [('A', 10), ('B', 5)]
    n_in = 4

    input = torch.zeros(1, n_in)
    lin = LinearDictOut(n_in, outputs)
    out = lin(input)

    for key, num in outputs:
        assert out[key].size(-1) == num

    out['A'][0, 0].backward()


@pytest.mark.parametrize('n, m', [(10, 5), (10, 10)])
def test_LinearFixed(n, m):
    def identity(x):
        return x[:, :m]

    mod = LinearFixed.from_affine(identity, n)
    a = torch.rand(1, n)
    # need to use the same dtype
    mod = mod.to(a.dtype)
    b = mod(a)
    assert b.shape == (1, m)
    return assert_tensors_allclose(identity(a), b)


def test_MapByKey():
    """Test that the module works"""
    name = 'a'
    n = 10

    class Mock(nn.Module):
        in_features = n

        def forward(self, x):
            return x

    funcs = {name: Mock()}
    data = {name: torch.rand(10)}
    mod = MapByKey(funcs)
    # test that forward evaluation works
    out = mod(data)
    out[name]  # out should be a dict
    assert isinstance(out, TensorDict)
    assert_tensors_allclose(out[name], data[name])

    # test inputs property
    assert mod.inputs == [(name, n)]
