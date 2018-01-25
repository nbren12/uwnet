import numpy as np
import xarray as xr
from xnoah.xcalc import centderiv
from xnoah.sam.coarsen import destagger
from .thermo import get_dz


def pad_along_axis(x, pad_width, axis=-1, **kwargs):
    pad_widths = [(0,0)]*x.ndim
    pad_widths[axis] = pad_width
    return np.pad(x, pad_widths, **kwargs)


def centered_diff(x, axis=-1, **kwargs):
    xpad = pad_along_axis(x, (1, 1), axis=axis, **kwargs)
    n = xpad.shape[axis]
    ip1= np.arange(2, n)
    im1= np.arange(0, n-2)
    return xpad.take(ip1, axis=axis) - xpad.take(im1, axis=axis)


def vertical_advection_upwind(w, phi, rho, dz, axis=-1):
    """Compute the vertical advection tendency using a first-order Godunov scheme

    F_i-1/2 = w_{i-1/2}^+ rho_{i-1} phi_{i-1} + (w_i-1/2)^- phi_{i} rho_{i}
    tend = -(F_{i+1/2} - F_{i-1/2})/dz_i / rho_i

    Assumes that z is the final dimension

    See Also
    --------
    http://www.noahbrenowitz.com/dokuwiki/doku.php?id=info:sam#arawaka_c-grid
    """
    n = w.shape[axis]

    i = np.arange(n)
    ip1 = np.r_[1:n, n - 1]
    im1 = np.r_[0, 0:n-1]

    wp = (w + np.abs(w)) / 2
    wm = (w - np.abs(w)) / 2

    flux = wm * rho * phi + wp * (phi * rho).take(im1, axis=axis)
    flux = pad_along_axis(flux, (0, 1), axis=-1, mode='constant')
    tend = np.diff(flux, axis=axis)/dz/rho

    return tend


def horizontal_advection(u, v, phi, dx=4000, dy=4000):
    """Assume

    Parameters
    ----------
    u : (*, y, x)
    v : (*, y, x)
    phi : (*, y, x)
    """

    fx = u*phi
    fy = v*phi

    x_adv = centered_diff(fx, axis=-1, mode='wrap')
    y_adv = centered_diff(fy, axis=-1, mode='edge')

    return x_adv/dx/2 + y_adv/dy/2


def xr_vert_adv(w, f, rho):
    dz = get_dz(rho.z)
    vert = xr.apply_ufunc(vertical_advection_upwind,
                          w, f, rho, dz,
                          input_core_dims=[['z'], ['z'], ['z'], ['z']],
                          output_core_dims=[['z']],
                          kwargs=dict(axis=-1))

    return vert


def xr_hor_adv(u, v, f):

    dy = dx = float(u.x[1] - u.x[0])

    # compute horizontal advection tendency
    dims = ['y', 'x']
    horiz = xr.apply_ufunc(horizontal_advection, u, v, f,
                           input_core_dims=[dims]*3,
                           output_core_dims=[dims],
                           kwargs=dict(dx=dx, dy=dy))

    return horiz
