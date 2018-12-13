import torch
from .modules import LinearDictIn, LinearDictOut, LinearFixed
import numpy as np

def assert_tensors_allclose(*args):
    args = [arg.detach().numpy() for arg in args]
    return np.testing.assert_allclose(*args)

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


def test_LinearFixed():
    def identity(x):
        return x

    n = 10
    mod = LinearFixed.from_affine(identity, n, n)
    a = torch.rand(n)
    # need to use the same dtype
    mod = mod.to(a.dtype)
    b = mod(a)
    return assert_tensors_allclose(a, b)
