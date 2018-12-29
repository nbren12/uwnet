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


def test_tensordict_needs_compatible_keys():
    a = TensorDict({'a': 1, 'c': 1})
    b = TensorDict({'a': 1})
    with pytest.raises(ValueError):
        a + b

def test_tensordict_copy():
    a = TensorDict({})
    assert isinstance(a.copy(), TensorDict)


def test_tensordict_apply():
    a = TensorDict({'a': 1})
    b = a.apply(lambda x: 2 * x)
    assert isinstance(b, TensorDict)
    assert b['a'] == 2


def test_tensordict_shape():
    n = 10
    shape = (1, n)
    a = TensorDict({'a': torch.ones(*shape)})
    assert a['a'].shape == shape
