import torch
from .jacobian import jacobian_backward, jacobian_norm, jacobian
import numpy as np
import pytest


def test_jacobian_backward():
    """Test the jacobian backprop function for a linear system

    y = A x

    For a linear system the gradient of Frobenious norm of the jacobian should
    be exactly equal to the original matrix.

    del_A  ||y_x||^2/2 = del_A || A ||^2  / 2 = A I = A


    """
    a = torch.rand(10, 10, requires_grad=True)
    x = torch.rand(10, requires_grad=True)

    y = a.matmul(x)
    jacobian_backward(y, x)

    A = a.grad.numpy()
    np.testing.assert_allclose(A, a.detach().numpy())


def test_jacobian_norm():

    a = torch.rand(10, 10, requires_grad=True)
    x = torch.rand(10, requires_grad=True)

    y = a.matmul(x)
    out = jacobian_norm(y, x)
    expected = a.norm() ** 2 / 2
    assert out.item() == pytest.approx(expected.item())

    # test gradient (see test_jacobian_backward docstring)
    out.backward()
    actual = a.grad.numpy()
    expected = a.data.numpy()
    np.testing.assert_allclose(actual, expected)


def test_jacobian():

    a = torch.rand(10, 10, requires_grad=True)
    x = torch.rand(10, requires_grad=True)

    y = a.matmul(x)
    out = jacobian(y, x)
    actual = out.data.numpy()
    expected = a.data.numpy()
    np.testing.assert_allclose(actual, expected)
