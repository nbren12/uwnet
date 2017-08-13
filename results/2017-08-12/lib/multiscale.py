"""Multiscale decomposition and IMMD equations
"""

import numpy as np
import scipy.fftpack as fft
import xarray as xr
from scipy.optimize import fsolve

from gnl.xlearn import XSVD
from gnl.xspline import Spline


def _get_lims(x):
    """Get limits of periodic dimension (if right end point is not included)

    """

    return float(x[0]), float(x[-1] * 2 - x[-2])


def chain_averages(A, avgs):
    for avg in avgs:
        A_new = avg(A)
        # perturbation
        yield (A - A_new).copy()
        A = A_new

    yield A


def fourier_smoothing_mat(n, a=.1, dof=None, p=6.0):
    """Low-pass filter

    This function returns the operator resulting from applying the the filter
        f(k) = 1/(1+a k^p)
    in spectral space.
    """

    I = np.eye(n)
    F = fft.fft(I, axis=-1)
    IF = fft.ifft(I, axis=-1)

    k = fft.fftfreq(n, d=1 / n)

    def filt(a, p):
        return np.diag(1 / (1 + np.exp(a) * np.abs(k)**p))

    if dof is None:
        N = filt(a, p)
        S = np.real(IF @ N @ F)
        return S

    else:

        def f(a):
            N = filt(a, p)
            return np.trace(N) - dof

        a = fsolve(f, 0)
        return fourier_smoothing_mat(n, a=a, dof=None)


class Averager(object):
    """Base clase for multiscale MESD/IPESD averaging

    This base-class uses cardinal splines for all averaging operations.
    """
    def __init__(self, domain, l_s=1.0):
        self.domain = domain
        self.l_s = l_s

    @classmethod
    def from_dataset(cls, A, **kwargs):

        domain = dict(x=_get_lims(A['x']), time=_get_lims(A['time']))
        return cls(domain, **kwargs)

    def _knots(self, dim, n=5, L=None):
        a, b = self.domain[dim]

        if L is not None:
            n = round((b - a) / L)

        return np.linspace(a, b, n)

    def _spline_avg(self, A, dim, bc, k=None, **kwargs):
        if k is None:
            k = self._knots(dim, **kwargs)
        return Spline(knots=k, dim=dim, bc=bc, order=4)\
            .fit(A)\
            .predict(A[dim])

    def planetary_t_avg(self, A):
        return self._spline_avg(A, 'time', 'extrap', L=5.0)

    def planetary_x_avg(self, A):
        return self._spline_avg(A, 'x', 'periodic', n=4)

    def synoptic_x_avg(self, A):
        return self._spline_avg(A, 'x', 'periodic', L=self.l_s)

    def synoptic_t_avg(self, A):
        return self._spline_avg(A, 'time', 'extrap', L=self.l_s)

    def synoptic_xt_avg(self, A):
        return A.pipe(self.synoptic_x_avg).pipe(self.synoptic_t_avg)

    def msdecomp(self, A):
        """Performs meso-scale synoptic scale decomposition"""

        if A.chunks is None:
            A = A.chunk(A.shape)

        avgs = [
            self.synoptic_xt_avg, self.planetary_x_avg, self.planetary_t_avg
        ]

        scales = ['m', 's', 'p', 'pb']
        scales = xr.DataArray(scales, dims='scale', name='scale',
                              coords={'scale': scales})
        return xr.concat(chain_averages(A, avgs), dim=scales)


class SVDAverager(Averager):
    def __init__(self, domain, adz=None, **kwargs):
        self.adz = adz
        super(SVDAverager, self).__init__(domain, **kwargs)

    @classmethod
    def from_dataset(cls, d):

        domain = dict(x=_get_lims(d['x']), time=_get_lims(d['time']))
        return cls(domain, d.adz)

    def planetary_x_avg(self, A, svd=None):

        if svd is None:
            svd = XSVD(feature_dims=['x', 'z'], weights=self.adz,
                       n_components=5)
            svd.fit(A)
            self._svd = svd

        # need to rename "sample" to "time"
        return svd.inverse_transform(svd.transform(A))\
                  .rename({'sample': 'time'})


def _mkxknots(x):
    L = float(x[-1]*2 - x[-2])
    L2 = L/2
    knots = [0, 1*L/4, L2 - L/8, L2 - L/16,
             L2, L2 + L/16, L2 + L/8, L * 3/4, L]
    return np.array(knots)


class AdaptiveAverager(Averager):

    def planetary_x_avg(self, A, svd=None):

        k = _mkxknots(A.x)
        return self._spline_avg(A, 'x', 'periodic', k=k)
