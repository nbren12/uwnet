from .tensordict import TensorDict
import torch
import pytest


def test_tensordict():
    b = {'a': torch.rand(1), 'c': torch.rand(1)}
    a = TensorDict(b)
    assert b.keys() == a.keys()

    # test dict
    for key in b:
        assert a[key] == b[key]

    # test arithmetic operations
    s = a + a
    assert isinstance(s, TensorDict)
    assert s['a'].item() == (a['a'] + a['a']).item()

    s = a / a
    assert isinstance(s, TensorDict)
    assert s['a'].item() == (a['a'] / a['a']).item()

    s = a * a
    assert s['a'].item() == (a['a'] * a['a']).item()

    assert s.keys() == a.keys()

    # test scalar ops
    assert isinstance(2 * a, TensorDict)
    assert (2 * a)['a'].item() == 2 * a['a'].item()
    assert (a * 2)['a'].item() == 2 * a['a'].item()

    # make sure this fails
    a = TensorDict({'a': 1, 'c': 1})
    b = TensorDict({'a': 1})
    with pytest.raises(ValueError):
        a + b
