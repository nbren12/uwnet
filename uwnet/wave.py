#!/usr/bin/env python
# coding: utf-8

# I want to develop the linearized 2D wave model described by Kuang (2018).
#
# Assuming that the mean winds are zero, the linearized anelastic equations are given by
#
# \begin{align}
# q_t +\bar{q}_z w = Q_2'\\
# s_t +\bar{s}_z w = Q_1'\\
# w_t  = -\partial x^2 L M B  -d w\\
# \end{align}
#
# $L$ and $M$ are operators, $Lf = -\frac{1}{\rho_0(z)}\int_0^z \rho_0 f dz$, and $Mf = + \int_0^z f dz $. The buoyancy $B$ is a function of $\bar{q} +q$ and $\bar{s} + s$. To constrain the constants, Kuang (2018) says that
#
# >For simplicity, I neglect the virtual effect in the large- scale wave dynamics, which was found to have only a minor contribution, and assume a rigid lid (w =0) at 175 hPa, as a radiating upper boundary condition is not essential for the instabilities that I shall consider.
#
#
# Kuang, Z. (2018). Linear stability of moist convecting atmospheres Part I: from linear response functions to a simple model and applications to convectively coupled waves. Journal of the Atmospheric Sciences. https://doi.org/10.1175/JAS-D-18-0092.1

# ## Derivation

# \begin{align}
# q_t +\bar{q}_z w = Q_2'\\
# s_t +\bar{s}_z w = Q_1'\\
# u_t + \phi_x =  - d u\\
# L \phi = B \\
# u_x + H w = 0\\
# \end{align}
#
# $Lf = f_z$ and $H w =\frac{1}{\rho_0} (\rho_0 w)_z$.
#
# $H w_t + [-d u - \phi_x]_x = 0$
#
# $H w_t + d H w - \phi_{xx} = 0$
#
# $L H w_t + d L H w -  B_{xx} = 0$
#
# $ LH (\partial_t + d ) W = B_{xx}$
#
#
# $LH =  \partial_z \frac{1}{\rho_0} \partial_z \rho_0 = A \rho$
#
# This can be discretized. We should have W collocated with S and Q to make the transport equations simpler. Then a suitable discretization of $A$ is given by
#
# $(A)_k v =\partial_z \frac{1}{\rho_0} \frac{v_k-v_{k-1}}{z^k-z^{k-1}}$
#
# $(A)_k v =\partial_z  \frac{v_k-v_{k-1}}{(z^k-z^{k-1})\rho^{k-1/2}}$
#
# $(A)_k v = \frac{1}{z_{k+1/2}-z_{k-1/2}} \left[ \frac{v_{k+1}-v_k}{(z_{k+1}-z_k)\rho^{k+1/2}} - \frac{v_k-v_{k-1}}{(z^k-z^{k-1})\rho^{k-1/2}} \right]$

# $(A v)_k = a_k v_{k-1} + b_k v_k + c_k v_{k+1}$
#
# $a_k = \frac{1}{(z_k - z_{k-1})(z_{k+1/2}-z_{k-1/2})\rho_{k-1/2}}$
#
# $b_k =- \frac{1}{(z_{k+1/2}-z_{k-1/2})}\left[  \frac{1}{(z_{k+1} - z_k)\rho_{k+1/2}} + \frac{1}{(z_k - z_{k-1})\rho_{k-1/2}} \right]$
#
# $c_k = \frac{1}{(z_{k+1} - z_k)(z_{k+1/2}-z_{k-1/2})\rho_{k+1/2}}$
#
# then
#
# $(LH w)_k =a_k \rho_{k-1} w_{k-1} + b_k \rho_k w_k + c_k \rho_{k+1} w_{k+1}$
#
# the dirichlet boundary conditions are satisfied by: $w_0 = - w_1 $ and $w_{n+1} = -w_n$. It is not simply $w_0$ because the vertical velocity should be located at the cell center. This shows up as
#
# $(LH w)_1 = - a_1 \rho_{0} w_{1} + b_1 \rho_1 w_1 + c_1 \rho_{2} w_{2} $
# and
#
# $(LH w)_n =a_n \rho_{n-1} w_{n-1} + b_n \rho_n w_n - c_n \rho_{n+1} w_{n}.$

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from toolz import curry

from collections import namedtuple
import torch
from src.data import open_data
from uwnet.jacobian import dict_jacobian
from uwnet.tensordict import TensorDict
from uwnet.thermo import interface_heights
from uwnet.utils import centered_difference


# Basic utilities and calculus
def xarray2torch(x):
    return torch.tensor(np.asarray(x), requires_grad=True).float()


def ones(n):
    """Convenience function"""
    return torch.ones(n, requires_grad=True)


def pad_linear(z):
    return np.r_[2 * z[0] - z[1], z, 2 * z[-1] - z[-2]]


def centered_to_interface(z):
    z = pad_linear(z)
    return (z[1:] + z[:-1]) / 2


def vertically_integrate(f, z_interface):
    dz = z_interface[1:] - z_interface[:-1]
    return torch.cumsum(f * dz, 0)


def deriv(f, z_center):
    return centered_difference(f, 0) / centered_difference(z_center, 0)


# Routines for computing elliptic operators
def get_elliptic_diagonals(rhoi, zi, rho, z):
    rho = pad_linear(rho)
    z = pad_linear(z)
    dz = np.diff(z)
    dzi = np.diff(zi)

    a = rho[:-2] / dz[:-1] / dzi / rhoi[:-1]
    b = -rho[1:-1] / dzi * (1 / dz[1:] / rhoi[1:] + 1 / dz[:-1] / rhoi[:-1])
    c = rho[2:] / dz[1:] / dzi / rhoi[1:]

    at = a[1:]
    bt = b.copy()
    bt[0] = b[0] - a[0] * (zi[0] - z[0]) / (z[1] - zi[0])
    bt[-1] = b[-1] - c[-1] * (z[-1] - zi[-1]) / (zi[-1] - z[-2])
    ct = c[:-1]

    return at, bt, ct


def get_elliptic_matrix(*args):
    diags = get_elliptic_diagonals(*args)
    A = np.asarray(sparse.diags(diags, [-1, 0, 1]).todense())
    return A


def get_elliptic_matrix_easy(rho, z):
    """Get elliptic matrix from cell centered rho and z"""
    rho = np.asarray(rho)
    z = np.asarray(z)

    rhoi = centered_to_interface(rho)
    zi = centered_to_interface(z)

    return get_elliptic_matrix(rhoi, zi, rho, z)


def dict_to_matrix(d, order):
    matrix = []
    for out_key in order:
        row = torch.cat(
            [d[out_key][in_key] for in_key in order], dim=-1)
        matrix.append(row)
    return torch.cat(matrix, dim=0).detach().numpy()


def subslice_blocks(jac, ind):
    out = jac.copy()
    for a in jac:
        for b in jac[a]:
            out[a][b] = jac[a][b][ind, ind]
    return out


def get_test_solution(base_state):
    from collections import namedtuple
    s, q = base_state['SLI'].detach(), base_state['QT'].detach()
    s.requires_grad = True
    q.requires_grad = True
    w = torch.zeros(s.size(0), requires_grad=True)
    soln = namedtuple('Solution', ['w', 's', 'q'])
    return soln(w=w, s=s, q=q)


class WaveEq:

    field_order = ('w', 's', 'q')

    def __init__(self, base_state, density, interface_heights,
                 center_heights):
        self.base_state = base_state
        self.density = density
        self.interface_heights = interface_heights
        self.center_heights = center_heights

    def buoyancy(self, s, q):
        s0 = self.base_state['SLI']
        return (s - s0) / s0

    @property
    def elliptic_operator(self):
        rho = self.density.detach().numpy()
        z = self.center_heights.detach().numpy()
        return get_elliptic_matrix_easy(rho, z)

    @property
    def inverse_elliptic_operator(self):
        return torch.tensor(np.linalg.inv(self.elliptic_operator)).float()

    def invert_buoyancy(self, s, q):
        b = self.buoyancy(s, q)
        return -torch.matmul(self.inverse_elliptic_operator, b)

    @property
    def sz(self):
        return deriv(self.base_state['SLI'], self.center_heights)

    @property
    def qz(self):
        return deriv(self.base_state['QT'], self.center_heights)

    def advection_s(self, w):
        return -self.sz * w

    def advection_q(self, w):
        return -self.qz * w

    def get_test_solution(self):
        return get_test_solution(self.base_state)

    def wave_matrix(self, k):
        soln = self.get_test_solution()
        outs = {
            's': self.advection_s(soln.w),
            'w': k**2 * self.invert_buoyancy(soln.s, soln.q),
            'q': self.advection_q(soln.w)
        }
        ins = soln._asdict()
        jac = dict_jacobian(outs, ins)
        jac = dict_to_matrix(jac, self.field_order)
        return jac


class WaveCoupler:

    def __init__(self, wave, src, base_state):
        self.wave = wave
        self.source_fn = src
        self.base_state = base_state

    @property
    def field_order(self):
        return self.wave.field_order

    def system_matrix(self, k):
        return self.wave.wave_matrix(k) + self.source_jacobian()

    def get_test_solution(self):
        return get_test_solution(self.base_state)

    def source_jacobian(self):

        sol = self.get_test_solution()
        outs = {}
        if self.source_fn is not None:
            srcs = self.source_fn({
                'SLI': sol.s,
                'QT': sol.q,
                'SST': self.base_state['SST'],
                'SOLIN': self.base_state['SOLIN']
            })
            outs['s'] = srcs['SLI'] / 86400
            outs['q'] = srcs['QT'] / 86400

        ins = sol._asdict()
        jac = dict_jacobian(outs, ins)

        # fill in zeros
        jac['w'] = {}
        a = jac['s']['s']
        for key in self.field_order:
            jac['w'][key] = torch.zeros_like(a)

        return dict_to_matrix(jac, self.field_order)

    def plot_eigs_spectrum(self):
        k, As = self.matrices()
        eigs = np.linalg.eigvals(As) / k[:, np.newaxis]

        plt.plot(k, eigs.imag, '.')

    def get_eigen_pair(self, k):
        return np.linalg.eig(self.system_matrix(k))

    def matrices(self):
        k = np.r_[:64] / 1000e3
        As = [self.system_matrix(kk) for kk in k]
        return k, As


def wave_from_xarray(mean, src=None):
    """Wave problem from xarray dataset"""

    base_state = {}

    for key in ['SLI', 'QT', 'SOLIN', 'SST']:
        base_state[key] = xarray2torch(mean[key])

    density = xarray2torch(mean.rho).float()
    zint = torch.tensor(interface_heights(mean.z), requires_grad=True).float()
    zc = xarray2torch(mean.z)

    wave = WaveEq(
        base_state=base_state,
        density=density,
        interface_heights=zint,
        center_heights=zc)

    return WaveCoupler(wave, src, base_state)


def init_test_wave():
    r"""A test wave problem with s = s0 + cz and rho0 = 1.0

    Notes
    -----

    Let's test the results of this code on a simpler example with constant density, and $s= s_0 + c z$.

    $$s_t + c w = 0$$
    $$u_t + p_x = 0 $$


    $S exp(ikx)_t + c W exp(ikx) $

    $U  exp(ikx)_t + ik P =0 $

    $ ik U + W_z = 0 $ and $ P_z = s/s_0$

    $ -ik W_zzt - k^2 P_z = 0 $

    $  W_zzt + k^2  s/s0 = 0 $

    $ w_{zztt} -  k^2  c /s_0  w = 0$

    $(\partial z^2 \partial t^2 - k^2 \frac{c}{s_0}) w = 0$

    $ w= Z(z) A(t)$

    $ Z'' A'' = k^2 \frac{c}{s_0} Z A $

    $w = sin(\pi z /H)$

    for this mode:

    $(-\pi^2 / H^2 \partial_t^2 -  k^2 \frac{c}{s_0}) w = 0$

    So that (not sure where the sign error came from)

    $(\partial_t^2 +  k^2 \frac{H^2}{\pi^2} \frac{c}{s_0}) w = 0$

    this is a wave equaiton with speed $\frac{H}{\pi} \sqrt{\frac{c}{s_0}}$
    """
    n = 10
    d = 1000
    zc = torch.arange(n).float() * d + d / 2
    zint = torch.arange(n + 1).float() * d

    c = .005
    s0 = 300
    H = zint[-1]

    density = ones(n)

    base_state = {'SLI': zc * c + s0, 'QT': ones(n)}

    speed = H / np.pi * np.sqrt(c / s0)

    return WaveEq(
        base_state=base_state,
        density=density,
        interface_heights=zint,
        center_heights=zc), speed


def get_wave_from_training_data(src=None):
    ds = open_data('training')
    mean = ds.isel(y=32, time=slice(0, 10)).mean(['x', 'time'])
    return wave_from_xarray(mean, src)

def get_base_state_from_training_data():
    ds = open_data('training')

    mean = ds.isel(y=32, time=slice(0, 10)).mean(['x', 'time'])
    base_state = {}

    for key in ['SLI', 'QT', 'SOLIN', 'SST']:
        base_state[key] = xarray2torch(mean[key])
    return base_state


def _expand_horiz_dims(d):
    out = {}
    for key in d:
        arr = d[key]
        n = arr.dim()
        if n == 1:
            out[key] = arr.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif n == 0:
            out[key] = arr.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError
    return out


@curry
def model_plus_damping(model, x, d0=1 / 86400.0):
    """Compute output of uwnet model with damping for vertical velocity

    Expands the appropriate dimensions of the inputs and outputs
    """

    d0 = 1 / 86400.0

    x = TensorDict(x)
    w = torch.zeros_like(x['SLI'])
    d = _expand_horiz_dims(x)
    inputs = TensorDict(d)
    outputs = model(inputs)
    outputs = outputs.apply(torch.squeeze)
    outputs['W'] = -w * d0

    return outputs


@curry
def marginize_model_over_solin(model, solin, inputs):
    """Marginalize out the diurnal cycle
    """
    outputs = []
    for sol in solin:
        inputs['SOLIN'] = torch.ones_like(inputs['SOLIN']) * sol
        outputs.append(model(inputs))
    return sum(outputs)/len(outputs)


def plot_struct_2d(w, z, n=256, **kwargs):
    """Plot structure of eigenfunction over one phase of oscillation
    """
    phase = 2*np.pi * np.r_[:n]/n
    phi = np.exp(1j * phase)[:,None]
    real_component = (w * phi).real
    plt.contourf(phase, z, real_component.T, **kwargs)
    plt.xlabel("phase")
    plt.ylabel("z (m)")
