import pytest
import torch

from .causal import (CausalLinearBlock, make_causal_mask,
                     make_causal_mask_from_ordinal)
from .testing import assert_tensors_allclose


def test_CausalLinearBlock():
    in_ordering = torch.tensor([1, 2, 3])
    out_ordering = torch.tensor([1, 1, 2, 2, 3, 3, 3])
    linear = CausalLinearBlock(in_ordering, out_ordering, activation=None)

    assert linear.in_features == len(in_ordering)
    assert linear.out_features == len(out_ordering)

    linear.weight

    x = torch.rand(1, 3, requires_grad=True)
    y = linear(x)
    assert y.size(-1) == len(out_ordering)

    # test dependency structure of module
    y[0, 0].backward()
    grad = x.grad[0, :]
    assert grad[-1].item() == pytest.approx(0)
    assert grad[0].item() != pytest.approx(0)

    x = torch.rand(1, 3, requires_grad=True)
    y = linear(x)
    y[0, 2].backward()
    grad = x.grad[0, :]
    assert grad[-1].item() == pytest.approx(0)
    assert grad[0].item() != pytest.approx(0)
    assert grad[1].item() != pytest.approx(0)


def test_make_causal_mask():
    n = 2
    dependency = [[0, 1], [0]]
    expected = torch.tensor([[1, 1], [1, 0]])

    out = make_causal_mask(n, dependency)
    assert_tensors_allclose(out, expected)


@pytest.mark.parametrize('input,output,expected,m', [
    ([0, 1], [0, 1], [[1, 1], [0, 1]], None),
    ([0, 1.0], [1.0, 1.0], [[1, 1], [1, 1]], None),
    ([1.0, 1.0], [1.0], [[1], [1]], .5),
    ([0.9, 1.0], [1.0], [[0], [1]], .5),
    ([0.4, 1.0], [1.0], [[1], [1]], .5),
])
def test_make_causal_mask_from_ordinal(input, output, expected, m):
    input = torch.tensor(input)
    output = torch.tensor(output)
    mask = make_causal_mask_from_ordinal(input, output, m)

    expected = torch.tensor(expected)

    assert_tensors_allclose(mask, expected)
