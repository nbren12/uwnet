import numpy as np
import pytest
from functools import partial
from unittest.mock import Mock

from .wave import (
    WaveCoupler,
    LinearResponseFunction,
    _fill_zero_above_input_level,
    filter_small_eigenvalues,
    get_elliptic_matrix,
    WaveEq,
)


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
    rhs = -c * np.pi * m / H ** 2 * np.cos(
        np.pi * m * z / H
    ) - np.pi ** 2 * m ** 2 / H ** 2 * np.sin(np.pi * m * z / H)

    # get matrix
    A = get_elliptic_matrix(rhoi, zi, rho, z)

    return A @ w, rhs


@pytest.mark.parametrize('test_soln', [
    partial(ans, m=3, H=1.0, c=5),
    partial(ans, m=3, H=1.0, c=0),
    partial(ans, m=1, H=16e3, c=0),
])
def test_elliptic_order_of_convergence(test_soln):

    err = []
    ns = [5, 10, 40, 100, 1000]
    for n in ns:
        pred, truth = test_soln(n=n)
        err.append(mad(pred, truth))
    assert order_of_convergence(ns, err) == pytest.approx(2.0, rel=0.1)


def test_numpy_linalg():
    """np.linalg.eig sometimes gives incorrect results

    This occured when using a docker image on the lambda box.
    """
    n = 10
    D = np.diag(np.arange(n))
    r, v = np.linalg.eig(D)
    I = np.eye(n)
    np.testing.assert_allclose(I, v)


def test_filter_small_eigenvalues():
    n = 10
    threshold = 4.5
    D = np.diag(np.arange(n))
    expected = np.where(D > threshold, D, 0.0)
    filtered = filter_small_eigenvalues(D, threshold)
    np.testing.assert_allclose(expected, filtered)


def create_jacobian(variables):
    # construct test data
    n = 5
    jacobian = {}
    for outkey in variables:
        jacobian[outkey] = {}
        for inkey in variables:
            jacobian[outkey][inkey] = np.ones((n, n))

    return jacobian


@pytest.mark.parametrize(
    "boundaries, variables",
    [({"a": 2, "b": 3}, ["a", "b"]), ({"a": 2, "b": 3}, ["a", "b", "c"])],
)
def test__fill_zero_above_input_level(boundaries, variables):
    jacobian = create_jacobian(variables)

    # call function
    ans = _fill_zero_above_input_level(jacobian, boundaries)

    # check answer
    for outkey in jacobian:
        for inkey in jacobian[outkey]:
            orig = jacobian[outkey][inkey]
            arr = ans[outkey][inkey]
            lid = boundaries.get(inkey, orig.shape[1])
            np.testing.assert_allclose(arr[:, lid:], 0.0)
            np.testing.assert_allclose(arr[:, :lid], orig[:, :lid])


def test_LinearResponseFunction_dump():
    from io import StringIO

    n = 10
    panes = {"s": {"s": np.random.rand(n, n)}}
    base = {"s": np.ones((n,))}
    lrf = LinearResponseFunction(panes, base)

    buffer = StringIO()
    lrf.dump(buffer)

    buffer.seek(0)
    lrf_loaded = LinearResponseFunction.load(buffer)

    for pane in lrf_loaded.iterpanes():
        assert isinstance(pane, np.ndarray)

    np.testing.assert_array_equal(lrf_loaded.panes["s"]["s"], lrf.panes["s"]["s"])


@pytest.fixture(params=[1e3, 5e3, 16e3])
def _test_wave(request):
    r"""A test wave problem with :math:`s = s_0 + cz` and :math:`\rho_0 = 1.0`


    Notes
    -----

    Let's test the results of this code on a simpler example with constant density, and
    :math:`s= s_0 + c z`.


    .. math::

        s_t + c w = 0\\
        u_t + p_x = 0 \\
        p_z = g \frac{s}{s_0}

    With some typical manipulations this can be written as one fourth order
    differential equation for w as follows:

    .. math::

        (\partial z^2 \partial t^2 + k^2 \frac{c g}{s_0}) w = 0

    The solution of this can be derived using separation of variables. Let
    :math:`w=Z(z)A(t)`, then

    .. math::

        Z'' A'' = k^2 \frac{c g}{s_0} Z A

    so that :math:`Z = sin(\pi z /H)`. for this mode:

    .. math::

        (\partial_t)^2 A + \lambda^2 A = 0 \\
        \lambda^2 = \frac{H^2 k^2 c g}{\pi^2 s_0}

    this is a wave equation with speed :math:`\lambda = \frac{H}{\pi} \sqrt{\frac{c}{s_0}}`
    """
    H = request.param

    n = 20
    d = H / n
    zc = np.arange(n) * d + d / 2
    zint = np.arange(n + 1) * d

    c = 0.005
    s0 = 300
    gravity = 9.81

    expected_speed = H / np.pi * np.sqrt(gravity * c / s0)

    density = np.ones(n)

    base_state = {
        "SLI": zc * c + s0,
        "QT": np.ones(n),
        "density": density,
        "height_interface": zint,
        "height_center": zc,
    }

    base_state = {key: val.astype(float) for key, val in base_state.items()}

    # need to patch in a constant in height buoyancy for this idealized test case to work
    def mock_buoyancy(s, q):
        return 9.81 * s / 300

    wave = WaveEq(base_state)
    wave.buoyancy = mock_buoyancy

    return wave, expected_speed


def test_wave_eq_correct_speed_integration(_test_wave):
    # This test is too complicated. Need to create a mock WaveEq object
    wave, expected_speed = _test_wave
    mock_lrf = Mock()
    mock_lrf.to_array.return_value = 0
    coupler = WaveCoupler(wave, mock_lrf)


    k = 3.0
    val = np.linalg.eigvals(coupler.system_matrix(k))
    growth_rate, phase_speed = val.real, val.imag / k

    assert growth_rate == pytest.approx(0.0)
    assert phase_speed.max() == pytest.approx(expected_speed, rel=.002)
