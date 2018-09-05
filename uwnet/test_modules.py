import torch
from .modules import LinearDictIn, LinearDictOut


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
