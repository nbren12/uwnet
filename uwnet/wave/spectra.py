import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from . import common
# from src.data import open_data
from .. import thermo
from wave import *


BOX_COLOR = "lightblue"

class paths:
    all = "../../nn/NNAll/20.pkl"
    lower = "../../nn/NNLowerDecayLR/20.pkl"
    nostab = "../../nn/NNLowerNoStabPenalty/20.pkl"


def sortbyvalue(eig):
    cp = eig.value.imag
    gr = eig.value.real
    permutation = cp * 100 + gr
    return eig.sortby(permutation)


def get_eigen_pair_xarray(wave, k):
    A = wave.system_matrix(k)
    lam, r = np.linalg.eig(A)
    return xr.Dataset(
        {"value": (["m"], lam), "vector": (["f", "m"], r)}, coords={"k": k}
    )


def compute_spectrum(wave, long_wave_km=40e6, short_wave_km=100e3) -> xr.Dataset:
    high_freq = 2 * np.pi / short_wave_km
    low_freq = 2 * np.pi / long_wave_km

    k = np.linspace(low_freq, high_freq, 100)
    eigs = [get_eigen_pair_xarray(wave, kk) for kk in k]
    return xr.concat(eigs, dim="k")


def plot_struct_x(eig):

    cp = eig.value.imag / eig.k
    targ = 20
    i = np.abs(cp - targ).argmin()
    eig = eig.isel(m=i)
    plot_struct_eig(eig)


def plot_struct_eig(eig):

    z = eig["z"]
    w, s, q = np.split(eig.vector, 3)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

    a.set_title("W")
    im = plot_struct_2d(w.values, z, ax=a)
    plt.colorbar(im, ax=a, fraction=0.05)

    b.set_title("S")
    im = plot_struct_2d(s.values, z, ax=b)
    plt.colorbar(im, ax=b, fraction=0.05)

    c.set_title("Q")
    im = plot_struct_2d(q.values, z, ax=c)
    plt.colorbar(im, ax=c, fraction=0.05)

    cp = float(eig.value.imag / eig.k)
    gr = 86400 * float(eig.value.real)

    fig.suptitle(f"cp = {cp:.2f} m/s; gr = {gr:.2f} 1/d")


def plot_struct_eig_p(
    vec, sources, p, rho, w_range=(-1, 1), s_range=(-0.5, 0.5), q_range=(-0.5, 0.5)
):

    fig, axs = plt.subplots(
        1, 5, figsize=(8, 3.5), constrained_layout=True, sharey=True, sharex=True
    )

    axs[0].invert_yaxis()

    p = np.asarray(p)
    rho = np.asarray(rho)

    w, s, q = np.split(vec, 3)
    _, q1, q2 = np.split(sources * 86400, 3)

    # convert to mb/hr
    w = thermo.omega_from_w(w, rho) * 3600 / 100
    # normalize by maximum w
    scale = np.max(np.abs(w))
    w, s, q = w / scale, s / scale, q / scale

    def get_kwargs(range):
        return dict(
            cmap="RdBu_r",
            # levels=np.linspace(range[0], range[1], 11)
        )

    def plot_pane(ax, title, *args, **kwargs):
        im = plot_struct_2d(*args, ax=ax, **kwargs)
        cbar = plt.colorbar(im, ax=ax, location="bottom")
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=70)

        ax.set_title(title, loc='left')

    plot_pane(axs[0], r"a) $\omega$ (hPa/hr)", w, p, **get_kwargs(w_range))
    plot_pane(axs[1], "b) s (K)", s, p, **get_kwargs(s_range))
    plot_pane(axs[2], "c) q (g/kg)", q, p, **get_kwargs(q_range))
    plot_pane(axs[3], "d) Q1 (K/day)", q1, p, **get_kwargs(s_range))
    plot_pane(axs[4], "e) Q2 (g/kg/day)", q2, p, **get_kwargs(q_range))

    axs[0].set_ylabel("p (mb)")

    return fig


def most_unstable(eig, c=100):
    m = int(np.abs(eig.value - c).argmin())
    return eig.isel(m=m)


def add_boxes(ax):
    max_cp = 5
    max_gr = 0.0
    width = 1000
    height = 100

    kwargs = dict(color=BOX_COLOR, zorder=-1000.0)

    rect_right = plt.Rectangle((max_cp, max_gr), width, height, **kwargs)
    rect_left = plt.Rectangle((-max_cp-width, max_gr), width, height, **kwargs)

    ax.add_patch(rect_left)
    ax.add_patch(rect_right)

    ax.axhline(0.0, color='black', zorder=-500)


def scatter_spectra(eig, ax=None, symlogy=True, cbar=True, box=True, s=10):
    if ax is None:
        ax = plt.gca()

    cp = eig.value.imag / eig.k
    gr = eig.value.real * 86400

    cp, gr, k = [_.values.ravel() for _ in xr.broadcast(cp, gr, eig.k)]

    if box:
        add_boxes(ax)


    with plt.rc_context({'figure.dpi': 100}):
        im = ax.scatter(cp, gr, c=k, cmap="viridis", s=s, rasterized=True)
    ax.set_ylim(gr.min(), gr.max())
    if symlogy:
        ax.set_yscale("symlog", linthresh=0.1)
    # ax.set_xticks([-50,-25,0,25,50])
    ax.set_xlabel("Phase speed (m/s)")
    ax.set_ylabel("Growth rate (1/d)")

    return im


def get_spectra_for_path(path):
    wave, mean = get_coupler(path)
    eig = compute_spectrum(wave)
    eig["z"] = mean.z
    eig["p"] = mean.p
    return eig


def get_mean_data():
    return open_data("mean").isel(y=32)


def get_coupler(path, **kwargs):
    mean = get_mean_data()
    model = torch.load(path)
    src_original = model_plus_damping(model, d0=1 / 86400 / 5)
    coupler = WaveCoupler.from_xarray_and_source(mean, source=src_original, **kwargs)
    return coupler, mean


def get_data():
    #     eig_no_penalty = get_spectra_for_path("../models/280/5.pkl")
    eig_unsable = get_spectra_for_path(paths.all)
    eig = get_spectra_for_path(paths.lower)
    no_pen = get_spectra_for_path(paths.nostab)

    return (eig, eig_unsable, no_pen)


def plot_compare_stability_models(data=None):
    if data is None:
        data = get_data()
    stable, unstable, no_penalty = data
    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(common.width, 3), sharey=True, sharex=True
    )

    scatter_spectra(stable.isel(k=slice(0, 32)), ax=a)
    plt.ylim(-2, 1)
    a.set_title("a) NO upper atmospheric input;\n with penalty")

    scatter_spectra(unstable, ax=b)
    b.set_title("b) upper atmospheric input;\n with penalty")

    scatter_spectra(no_penalty, ax=c)
    c.set_title("c)No upper atmosphere\n no penalty")

    return data


def find_eig(values, k, phase_speed, growth_rate):
    query = -phase_speed * 1j * k + growth_rate / 86400
    closest_index = np.argmin(np.abs(values - query))
    return closest_index


def plot_structure(coupler, p, rho, k, phase_speed, growth_rate, **kwargs):
    # get closest eigenvector
    values, vectors = coupler.get_eigen_pair(k)

    # get the matching eigenvector and value
    closest_index = find_eig(values, k, phase_speed, growth_rate)
    vector = vectors[:, closest_index]
    value = values[closest_index]

    sources = coupler.source_terms(vector)

    # plot the eigenpair
    fig = plot_struct_eig_p(vector, sources, p, rho, **kwargs)
    cp = -value.imag / k
    gr = value.real * 86400
    wave_length = 2 * np.pi / k / 1000
    plt.suptitle(
        rf"$\sigma$ = {gr:.2f} 1/d; $c_p$ ={cp:.2f} m/s; $\lambda$={wave_length:.0f} km"
    )
    return fig


def get_linear_response_functions():
    wave_all = get_coupler(paths.all)[0].lrf
    wave_lower = get_coupler(paths.lower)[0].lrf
    wave_nostab = get_coupler(paths.nostab)[0].lrf

    return wave_all, wave_lower, wave_nostab


def plot_lrf_eigs(lrf, ax=None):
    variables = ["s", "q"]
    A = lrf.to_array(variables)
    values = np.linalg.eigvals(A) * 86400

    ax.plot(values.real, values.imag, ".")
    ax.grid()


def plot_compare_models_lrfs():
    lrfs = get_linear_response_functions()
    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(4, 6))
    titles = ["All", "Lower", "LowerNoStab"]

    for k in range(len(axs)):
        plot_lrf_eigs(lrfs[k], axs[k])
        axs[k].set_title(titles[k])
        axs[k].set_ylabel(r"$\Im$ (1/day)")

    axs[-1].set_xlabel(r"$\Re $(1/day)")


def spectra_report(
    model_path, xlim_zoom=(-1, 1), structure_plots=((1.0, 0.2), (0, 0.9)), lrf_lid=None
):
    coupler, mean = get_coupler(model_path, lrf_lid=lrf_lid)
    eig = compute_spectrum(coupler)

    with plt.style.context("ggplot"):
        scatter_spectra(eig)
        plt.figure()
        scatter_spectra(eig, symlogy=False)
        plt.xlim(xlim_zoom)
        plt.ylim(bottom=0)

        for k, cp, gr in structure_plots:
            plt.figure()
            plot_structure(coupler, mean.p, mean.rho, k, phase_speed=cp, growth_rate=gr)


def plot_structure_from_path(model_path, structure=None, lrf_lid=None):
    coupler, mean = get_coupler(model_path, lrf_lid=lrf_lid)

    eig = compute_spectrum(coupler)
    k, cp, gr = structure
    return plot_structure(coupler, mean.p, mean.rho, k, phase_speed=cp, growth_rate=gr)


# TODO: refactor these plotting routines
def plot_struct_2d(w, z, n=256, ax=None, **kwargs):
    """Plot structure of eigenfunction over one phase of oscillation
    """
    phase = 2 * np.pi * np.r_[:n] / n
    phi = np.exp(1j * phase)[:, None]
    real_component = (w * phi).real
    im = ax.contourf(phase, z, real_component.T, **kwargs)
    ax.set_xlabel("phase (rad)")
    return im
