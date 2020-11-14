from .tensordict import TensorDict, stack, lag_tensor
from .testing import assert_tensors_allclose
import torch
import pytest


def _get_tensordict_example():
    n = 10
    shape = (1, n)
    return TensorDict({'a': torch.ones(*shape)})


def _get_tensordict_from_shapes(shape1, shape2):
    return TensorDict({'a': torch.rand(shape1), 'c': torch.rand(shape2)})


@pytest.mark.parametrize('shape1, shape2, dim, raises_error', [
    ([4], [4], 0, False),
    ([3], [4], 0, True),
    ([1, 3], [1, 3], 0, False),
    ([1, 3], [1, 4], 0, False),
    ([1, 3], [1, 3], 1, False),
])
def test_tensordict_size(shape1, shape2, dim, raises_error):
    d = _get_tensordict_from_shapes(shape1, shape2)
    if raises_error:
        with pytest.raises(ValueError):
            d.size(dim)
    else:
        assert d.size(dim) == shape1[dim]


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


def test_tensordict_split():
    n = 10
    shape = (1, n)
    a = TensorDict({'a': torch.ones(*shape)})

    splits = a.split(1, dim=1)

    for split in splits:
        assert isinstance(split, TensorDict)
        # test that key is present
        split['a']
        assert split['a'].shape == (1, 1)


tensor = _get_tensordict_example()['a']
@pytest.mark.parametrize('attr', dir(tensor))
def test_tensordict_dispatch(attr):
    if attr in ['imag', 'real']:
        pytest.xfail()
    t = _get_tensordict_example()
    a = t['a']
    getattr(t, attr)


def test_stack():
    t = _get_tensordict_example()
    n_stack = 2
    shape = t['a'].shape
    out = stack([t] * n_stack, dim=0)
    assert isinstance(out, TensorDict)
    out['a']
    assert out['a'].shape == (n_stack,) + shape


def test_lag_tensor():
    a = torch.arange(10)
    b = lag_tensor(a, 1, 0)
    assert_tensors_allclose(a[1:], b)

    b = lag_tensor(a, -1, 0)
    assert_tensors_allclose(a[:-1], b)
