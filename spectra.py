
from uwnet.wave import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from uwnet import thermo
from src.data import open_data
import common

class paths:
    all = "../../nn/NNAll/20.pkl"
    lower = "../../nn/NNLowerDecayLR/20.pkl"
    nostab = "../../nn/NNLowerNoStabPenalty/20.pkl"


def sortbyvalue(eig):
    cp = eig.value.imag
    gr = eig.value.real
    permutation = cp *100 + gr
    return eig.sortby(permutation)


def get_eigen_pair_xarray(wave, k):
    lam, r = wave.get_eigen_pair(k=k)
    return xr.Dataset({'value': (['m'], lam), 'vector':(['f', 'm'], r)}, coords={'k': k})


def get_spectrum(wave):
    k = 2*np.pi * np.r_[:128] / 40e6
    eigs = [get_eigen_pair_xarray(wave, kk) for kk in k]
    return xr.concat(eigs, dim='k')


def get_sorted_spectrum(wave):
    eig = get_spectrum(wave)
    return eig.groupby('k').apply(sortbyvalue)


def plot_struct_x(eig):

    cp = eig.value.imag/eig.k
    targ = 20
    i = np.abs(cp - targ).argmin()
    eig = eig.isel(m=i)
    plot_struct_eig(eig)
    
    
def plot_struct_eig(eig):

    z = eig['z']
    w, s, q = np.split(eig.vector, 3)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    
    a.set_title('W')
    im = plot_struct_2d(w.values, z, ax=a)
    plt.colorbar(im, ax=a, fraction=.05)

    b.set_title('S')
    im = plot_struct_2d(s.values, z, ax=b)
    plt.colorbar(im, ax=b, fraction=.05)

    c.set_title('Q')
    im = plot_struct_2d(q.values, z, ax=c)
    plt.colorbar(im, ax=c, fraction=.05)
    
    cp = float(eig.value.imag/eig.k)
    gr = 86400 * float(eig.value.real)
    
    fig.suptitle(f"cp = {cp:.2f} m/s; gr = {gr:.2f} 1/d")

    
def plot_struct_eig_p(vec, p):

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), 
                            constrained_layout=True, sharey=True, sharex=True)

    axs[0].invert_yaxis()

    w, s, q = np.split(vec, 3)
    im = plot_struct_2d(w, p, ax=axs[0])
    plt.colorbar(im, ax=axs[0], fraction=.1)
    axs[0].set_title('W')

    im = plot_struct_2d(s, p, ax=axs[1])
    plt.colorbar(im, ax=axs[1])
    axs[1].set_title('s')

    im = plot_struct_2d(q, p, ax=axs[2])
    plt.colorbar(im, ax=axs[2])
    axs[2].set_title('q');
    
    axs[0].set_ylabel('p (mb)')
    
def most_unstable(eig, c=100):
    m = int(np.abs(eig.value-c).argmin())
    return eig.isel(m=m)


def scatter_spectra(eig, ax=None):
    if ax is None:
        ax = plt.gca()
        
    cp = eig.value.imag/eig.k
    gr = eig.value.real * 86400

    cp, gr, k = [_.values.ravel() for _ in xr.broadcast(cp, gr, eig.k)]

    ax.scatter(cp , gr, c=plt.cm.Blues(k/k.max()), s=.5)
    ax.set_ylim(gr.min(), gr.max())
    ax.set_yscale('symlog', linthreshy=.1)
    ax.set_xticks([-50,-25,0,25,50])
    ax.grid()
    
def get_spectra_for_path(path):
    wave, mean = get_coupler(path)
    eig = get_spectrum(wave)
    eig['z'] = mean.z
    eig['p'] = mean.p
    return eig


def get_mean_data():
    data = open_data("training").sel(time=slice(110, 120)).load()
    mean = data.mean(['x', 'time']).isel(y=32)
    return mean


def get_coupler(path):
    mean = get_mean_data()
    model = torch.load(path)
    src_original = model_plus_damping(model,d0=1/86400/5)
    return WaveCoupler.from_xarray_and_source(mean, source=src_original, lrf_lid={'s': 20, 'q': 20}), mean

    
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
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(common.width, 3), sharey=True, sharex=True)
    
    scatter_spectra(stable.isel(k=slice(0,32)), ax=a)
    plt.ylim(-2, 1)
    a.set_title("a) NO upper atmospheric input;\n with penalty");
    
    scatter_spectra(unstable, ax=b)
    b.set_title("b) upper atmospheric input;\n with penalty");
    
    scatter_spectra(no_penalty, ax=c)
    c.set_title('c)No upper atmosphere\n no penalty')
    
    
    return data
    
    
def plot_structure(coupler, p, wave_length, phase_speed, growth_rate):
        # get closest eigenvector
    k = 2*np.pi/wave_length
    A = coupler.system_matrix(k)
    values, vectors = np.linalg.eig(A)
    
    # get the matching eigenvector and value
    query = -phase_speed * 1j * k + growth_rate / 86400
    closest_index = np.argmin(np.abs(values-query))
    vector = vectors[:, closest_index]
    value = values[closest_index]
    
    # plot the eigenpair
    plot_struct_eig_p(vector, p)
    cp = -value.imag/k
    gr = value.real *86400
    plt.suptitle(f"Gr = {gr:.2f}; Cp={cp:.2f} ")
    
    
def plot_structure_nnlower_unstable_mode(**kwargs):
    path = "../../nn/NNLowerDecayLR/20.pkl"
    coupler, mean = get_coupler(path)
    p = np.array(mean.p)
    plot_structure(coupler, p, **kwargs)

    
def plot_structure_nnall_unstable_mode(**kwargs):
    path = "../../nn/NNAll/20.pkl"
    coupler, mean = get_coupler(path)
    p = np.array(mean.p)
    plot_structure(coupler, p, **kwargs)

    
def get_linear_response_functions():
    wave_all =  get_coupler(paths.all)[0].lrf
    wave_lower =  get_coupler(paths.lower)[0].lrf
    wave_nostab =  get_coupler(paths.nostab)[0].lrf
    
    return wave_all, wave_lower, wave_nostab


def plot_lrf_eigs(lrf, ax=None):
    variables = ['s', 'q']
    A = lrf.to_array(variables)
    values = np.linalg.eigvals(A) * 86400

    ax.plot(values.real, values.imag, '.')
    ax.grid()
    

def plot_compare_models_lrfs():
    lrfs = get_linear_response_functions()
    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(4, 6))
    titles = ['All', 'Lower', 'LowerNoStab']

    for k in range(len(axs)):
        plot_lrf_eigs(lrfs[k], axs[k])
        axs[k].set_title(titles[k])
        axs[k].set_ylabel(r'$\Im$ (1/day)')
        
    axs[-1].set_xlabel(r'$\Re $(1/day)')