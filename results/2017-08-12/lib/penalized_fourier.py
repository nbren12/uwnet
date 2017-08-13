"""Fourier smoothing filter
"""
import numpy as np
from scipy.optimize import fsolve
import scipy.fftpack as fft

def compute_penal(n, dof, p=6.0):
    """Compute penalization coefficent of low-pass filter

    This function returns the operator resulting from applying the the filter
        f(k) = 1/(1+exp^a k^p)
    in spectral space.

    Parameters
    ----------
    n: int
        number of spatial grid points
    dof: int
        number of degrees of freedom
    p: float, optional
        power to be used in filter (default: p=6)

    Returns
    -------
    f: np.ndarray
         filter to apply in fourier space
    a: float
         penalization coefficent

    """

    I = np.eye(n)
    F = fft.fft(I, axis=-1)
    IF = fft.ifft(I, axis=-1)
    k = fft.fftfreq(n, d=1 / n)

    def filt(a, p):
        # use exp(a) for numerical purposes
        return 1 / (1 + np.exp(a) * np.abs(k)**p)

    def f(a):
        N = filt(a, p)
        return np.sum(N) - dof

    a = fsolve(f, 0)
    return filt(a,p), float(a)


