#!/usr/bin/env python
# coding: utf-8
import json
from collections import Mapping, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse
from toolz import curry
from toolz.curried import compose, first, get, valmap

from ..jacobian import dict_jacobian
from .tensordict import TensorDict
from ..thermo import Lc, cp, grav, interface_heights

from . import utils


def centered_difference(x, dim=-3):
    """Compute centered difference with first order difference at edges"""
    x = x.transpose(dim, 0)
    dx = torch.cat(
        [
            torch.unsqueeze(x[1] - x[0], 0), x[2:] - x[:-2],
            torch.unsqueeze(x[-1] - x[-2], 0)
        ],
        dim=0)
    return dx.transpose(dim, 0)


def get_test_solution(base_state):
    from collections import namedtuple

    s, q = base_state["SLI"].detach(), base_state["QT"].detach()
    s.requires_grad = True
    q.requires_grad = True
    w = torch.zeros(s.size(0), requires_grad=True)
    soln = namedtuple("Solution", ["w", "s", "q"])
    return soln(w=w, s=s, q=q)


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


# Functions for manipulating jacobian dictionaries #########################
def dict_to_matrix(d, order):
    matrix = []
    for out_key in order:
        row = np.concatenate([d[out_key][in_key] for in_key in order], axis=-1)
        matrix.append(row)
    return np.concatenate(matrix, axis=0)


def get_first_pane(jac):
    for i in jac:
        for j in jac[i]:
            return jac[i][j]


def ensure_panes_present(jac, required_keys):
    """Ensure that all required keys are present in an input dictionary"""
    output = {}
    # fill in zeros
    template = np.zeros_like(get_first_pane(jac))

    for i in required_keys:
        output[i] = {}
        for j in required_keys:
            try:
                output[i][j] = jac[i][j]
            except KeyError:
                output[i][j] = template
    return output


def _fill_zero_above_input_level(jac, boundaries):
    output = {}
    for outkey in jac:
        output[outkey] = {}
        for inkey in jac[outkey]:
            arr = jac[outkey][inkey]
            upper_lid = boundaries.get(inkey, arr.shape[1])
            i, j = np.ogrid[[slice(n) for n in arr.shape]]
            output[outkey][inkey] = np.where(j < upper_lid, arr, 0.0)
    return output


def subslice_blocks(jac, ind):
    out = jac.copy()
    for a in jac:
        for b in jac[a]:
            out[a][b] = jac[a][b][ind, ind]
    return out


def filter_small_eigenvalues(matrix: np.array, threshold: float, replace: float = 0):
    s, v = np.linalg.eig(matrix)
    s_filt = np.where(np.abs(s) > threshold, s, replace)
    return (v * s_filt) @ np.linalg.inv(v)
    # u, s, vh = np.linalg.svd(matrix)
    # s_filt = np.where(s> threshold, s, replace)
    # return (u * s_filt) @ vh


class WaveEq:

    field_order = ("w", "s", "q")

    def __init__(self, base_state):
        """

        Parameters
        ----------
        base_state
            dict of numpy arrays. Must have the following keys:
                'SLI', 'QT', 'density', 'height_center'
        """
        self.base_state = base_state
        self._base_state = utils.numpy2torch(base_state)

    @property
    def density(self) -> torch.Tensor:
        return self._base_state["density"]

    @property
    def center_heights(self) -> torch.Tensor:
        return self._base_state["height_center"]

    def buoyancy(self, s, q):
        grav = 9.81
        s0 = self._base_state["SLI"]
        return grav * (s - s0) / s0

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
        return deriv(self._base_state["SLI"], self.center_heights)

    @property
    def qz(self):
        return deriv(self._base_state["QT"], self.center_heights)

    def advection_s(self, w):
        return -self.sz * w

    def advection_q(self, w):
        return -self.qz * w

    def get_test_solution(self):
        return get_test_solution(self._base_state)

    def system_matrix(self, k):
        soln = self.get_test_solution()
        outs = {
            "s": self.advection_s(soln.w),
            "w": k ** 2 * self.invert_buoyancy(soln.s, soln.q),
            "q": self.advection_q(soln.w),
        }
        ins = soln._asdict()
        jac = dict_jacobian(outs, ins)
        jac = utils.torch2numpy(jac)
        jac = dict_to_matrix(jac, self.field_order)
        return jac

    @staticmethod
    def from_tom_base_state(tom_base_state):
        """Return WaveEq from Tom's base state"""
        # TODO refactor this to an abstract factor (A)
        tom_base_state = valmap(compose(np.copy, np.flip), tom_base_state)
        q, T, z, rho = get(["qv", "T", "z", "rho"], tom_base_state)
        base_state = {"QT": q * 1000.0, "SLI": T + grav / cp * z, "height_center": z, "density": rho}
        return WaveEq(base_state)


def tom_base_state_to_base_state(tom_base_state):
        tom_base_state = valmap(compose(np.copy, np.flip), tom_base_state)
        q, T, z, rho = get(["qv", "T", "z", "rho"], tom_base_state)
        return {"QT": q * 1000.0, "SLI": T + grav / cp * z, "height_center": z, "density": rho}


def tom_jacobian_to_jacobian(panes):
    jac = defaultdict(dict)

    def flip_axes_and_squeeze(x):
        return x[::-1, ::-1].squeeze()

    q_scale = Lc / 1000
    t_scale = cp

    jac["q"]["q"] = panes["q"]["q"]
    jac["q"]["s"] = panes["q"]["T"] / (q_scale / t_scale)
    jac["s"]["q"] = panes["T"]["q"] / (t_scale / q_scale)
    jac["s"]["s"] = panes["T"]["T"]

    jac = valmap(valmap(flip_axes_and_squeeze), jac)

    return jac


class NumpyEncoder(json.JSONEncoder):
    """From stackoverflow:
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class LinearResponseFunction:
    """Linearized Response Function in (SLI, QT) space

    Attributes
    ----------
    panes
        dictionary of the linearized response function.
        panes[response][input][i,j] is the sensitivity of response[i] to
        input[j]. This data using the following physical units:

            panes['q']['q'] : g/kg / (g/kg) / s
            panes['q']['s'] : g/kg / K / s
            panes['s']['q'] : K / (g / kg) / s
            panes['s']['s'] : K / K / s

    """

    panes: Dict[str, Dict[str, np.array]]
    base_state: Dict[str, np.array]

    def iterpanes(self):
        for i in self.panes:
            for j in self.panes[i]:
                yield self.panes[i][j]

    def first_pane(self):
        first_row = first(self.panes.values())
        return first(first_row.values())

    def size(self):
        return self.first_pane().shape[0]

    def dump(self, file):
        return json.dump(asdict(self), file, cls=NumpyEncoder)

    @staticmethod
    def load(file):
        spec = json.load(file)
        panes = spec.pop("panes")

        new_panes = {}
        for i in panes:
            new_panes[i] = {}
            for j in panes[i]:
                new_panes[i][j] = np.asarray(panes[i][j])
        return LinearResponseFunction(new_panes, **spec)

    def to_array(self, field_order):
        jac = ensure_panes_present(self.panes, field_order)
        return dict_to_matrix(jac, field_order)


@dataclass
class FilteredLinearResponseFunction(LinearResponseFunction):

    threshold: float = 1.0
    replace: float = 0.0

    def to_array(self, field_order):
        A = super(FilteredLinearResponseFunction, self).to_array(field_order)
        return filter_small_eigenvalues(A, self.threshold, self.replace)



def ablate_upper_atmosphere(jacobian, boundaries):
    return _fill_zero_above_input_level(jacobian, boundaries)

@dataclass
class LowerAtmosLRF:
    """Linearized Response Function with no upper atmospheric sensitivity

    Attributes
    ----------
    boundaries
        dict of upper level boundaries
    lrf
        linear response function argument. Must have a `panes` attribute.
    """

    boundaries: Mapping
    pane: LinearResponseFunction

    def to_array(self, field_order):
        jacobian = ensure_panes_present(self.lrf.panes, field_order)
        jacobian = _fill_zero_above_input_level(jacobian, self.boundaries)
        return dict_to_matrix(jacobian, field_order)


@dataclass
class WaveCoupler:
    """Couple Linearized Response Function to Waves

    Attributes
    ----------
    wave : WaveEq
    lrf : LinearizedResponseFunction
    base_state : dict of numpy arrays
    """

    def __init__(self, wave: WaveEq, lrf: LinearResponseFunction):
        self.wave = wave
        self.lrf = lrf

    @property
    def base_state(self):
        return self.wave.base_state

    @property
    def field_order(self):
        return self.wave.field_order

    def system_matrix(self, k):
        return self.wave.system_matrix(k) + self.source_jacobian()

    def source_jacobian(self):
        return self.lrf.to_array(self.field_order)

    def plot_eigs_spectrum(self):
        k, As = self.matrices()
        eigs = np.linalg.eigvals(As) / k[:, np.newaxis]

        plt.plot(k, eigs.imag, ".")

    def get_eigen_pair(self, k):
        return np.linalg.eig(self.system_matrix(k))

    def matrices(self):
        k = np.r_[:64] / 1000e3
        As = [self.system_matrix(kk) for kk in k]
        return k, As

    def source_terms(self, vec):
        return self.source_jacobian() @ vec

    @staticmethod
    def from_tom_data(tom_data):
        # TODO refactor this to an abstract factor (A)
        base_state = tom_base_state_to_base_state(tom_data['base_state'])
        jacobian = tom_jacobian_to_jacobian(tom_data["jacobian"])
        wave_eq = WaveEq(base_state)
        lrf =  LinearResponseFunction(panes=jacobian, base_state=base_state)
        return WaveCoupler(wave_eq, lrf)


def base_from_xarray(mean):
    base_state = {}
    for key in ["SLI", "QT", "SOLIN", "SST"]:
        base_state[key] = utils.xarray2numpy(mean[key])

    base_state["density"] = utils.xarray2numpy(mean.rho)
    base_state["height_interface"] = interface_heights(mean.z)
    base_state["height_center"] = utils.xarray2numpy(mean.z)
    base_state["pressure"] = utils.xarray2numpy(mean.p)
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
    w = torch.zeros_like(x["SLI"])
    d = _expand_horiz_dims(x)
    inputs = TensorDict(d)
    outputs = model(inputs)
    outputs = outputs.apply(torch.squeeze)
    outputs["W"] = -w * d0

    return outputs


# TODO Delete this function if it isn't used
@curry
def marginize_model_over_solin(model, solin, inputs):
    """Marginalize out the diurnal cycle
    """
    outputs = []
    for sol in solin:
        inputs["SOLIN"] = torch.ones_like(inputs["SOLIN"]) * sol
        outputs.append(model(inputs))
    return sum(outputs) / len(outputs)



def plot_eigvals(A):
    vals = np.linalg.eigvals(A)
    plt.plot(vals.real, vals.imag, ".")
    plt.xlabel(r"$\Re$")
    plt.ylabel(r"$\Im$")


def plot_phase_speed(A, k):
    vals = np.linalg.eigvals(A)
    plt.plot(vals.real * 86400, vals.imag / k, ".")
    plt.xlabel("Growth rate (1/day)")
    plt.ylabel("Phase speed (m/s)")
