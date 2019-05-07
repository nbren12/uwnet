import numpy as np
import pytest

from uwnet.wave import get_elliptic_matrix


def mad(x, y):
    return np.abs(x - y).mean()


def order_of_convergence(n, err):
    return -np.polyfit(np.log(n), np.log(err), 2)[1]


def ans(m, H, c=2, n=32):
    """Method of manufactured solutions test"""
    # setup grid
    d = H / n
    z = np.arange(n) * d + d / 2
    zi = np.arange(n + 1) * d

    # setup solution
    rho = np.exp(-c * z / H)
    rhoi = np.exp(-c * zi / H)
    #     rhoi = centered_to_interface(rho)
    w = np.sin(m * np.pi * z / H)

    # setup rhs
    rhs = -c * np.pi * m / H**2 * np.cos(
        np.pi * m * z / H) - np.pi**2 * m**2 / H**2 * np.sin(
            np.pi * m * z / H)

    # get matrix
    A = get_elliptic_matrix(rhoi, zi, rho, z)

    return A @ w, rhs


def test_elliptic_order_of_convergence():

    err = []
    ns = [5, 10, 40, 100, 1000]
    for n in ns:
        pred, truth = ans(3, 1.0, c=5, n=n)
        err.append(mad(pred, truth))
    assert order_of_convergence(ns, err) == pytest.approx(2.0, rel=.1)
