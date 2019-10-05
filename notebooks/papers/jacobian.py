# compute boot strap stats
from collections import defaultdict
from random import randint

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
import matplotlib
import numpy as np
import xarray as xr
from toolz import curry

plt.rcParams['hatch.color'] = '0.5'

import torch
from common import get_vmax
from src.data import ngaqua_climate_path
from uwnet.jacobian import jacobian, jacobian_from_model
from uwnet.thermo import compute_apparent_source
from uwnet.xarray_interface import dataset_to_torch_dict


def saliency_map_one_location(model, ds):
    torch_data = dataset_to_torch_dict(ds)
    torch_data = torch_data[model.input_names]

    # compute saliency map
    return jacobian_from_model(model, torch_data)


def plot(args):
    """Plot jacobian

    Parameters
    ----------
    args : tuple
        (jac, p) tuple
    """
    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(5, 4))
    jac, p = args

    k = 0

    abc = 'abcd'

    progs = list(jac)

    for i, outkey in enumerate(progs):
        for j, inkey in enumerate(progs):
            ax = axs[i, j]
            val = jac[outkey][inkey]
            val = val.detach().numpy()

            # select the colorbar norm
            vmax = get_vmax(val)
#             do_sym_log = vmax > 10
           
#             if do_sym_log:
#                 linthresh = 1.0
#                 norm = SymLogNorm(
#                     linthresh=linthresh, linscale=20, vmin=-vmax, vmax=vmax)
#             else:
#                 norm = Normalize(vmin=-vmax, vmax=vmax)
            if inkey == 'QT':
                norm = Normalize(vmin=-1.5, vmax=1.5)
            else:
                norm =  Normalize(vmin=-1.0, vmax=1.0)
            
            im = ax.pcolormesh(p, p, val, cmap='RdBu_r', norm=norm)
            ax.contourf(
                p, p, val, levels=[-100,-10,10, 100] , extend='both', colors='none',
                hatches=['xxxx', '...', None, '...', 'xxxx'],)
            
            letter = abc[k]
            ax.set_title(f"{letter}) d{outkey}/dt from {inkey}", loc='left')
            ax.set_xlabel(f"Pressure ({inkey})")
            ax.set_ylabel(f"Pressure (d{outkey}/dt)")
            ax.invert_yaxis()
            ax.invert_xaxis()
    
            cb = plt.colorbar(im, ax=ax, pad=-.04)
            
#             if do_sym_log:
#                 locator = matplotlib.ticker.SymmetricalLogLocator(
#                     linthresh=linthresh, base=10)
#                 locator.set_params(numticks=5)
#                 cb.locator = locator
#                 cb.update_ticks()

            k += 1

    return axs


def get_model(path="../../models/265/5.pkl"):
    # open model
    model = torch.load(path)
    model.eval()

    return model


def bootstrap_samples(tropics, n):
    sample_dims = ['time', 'x', 'y']
    dim_name = 'sample'
    indexers = {
        dim: xr.DataArray(np.random.choice(tropics[dim], n), dims=[dim_name])
        for dim in sample_dims
    }
    samples_dataset = tropics.sel(**indexers)

    for i in range(n):
        rand_ind = randint(0, n - 1)
        sample = (samples_dataset.isel(sample=rand_ind)
                  .expand_dims(['y', 'x'], [-2, -1]).compute())
        yield sample


def get_jacobian(model, sample):
    necessary_variables = sample[model.input_names]
    jac = saliency_map_one_location(model, necessary_variables)
    return jac


def apply_list_jacobian(func, seq):
    keys = seq[0].keys()
    output = defaultdict(dict)
    for ink in keys:
        for outk in keys:
            output[outk][ink] = func([it[outk][ink] for it in seq])
    return output


# Boot strap statistics
def mean(seq):
    n = len(seq)
    return sum(seq) / n


def std(seq):
    n = len(seq)
    mu = mean(seq)
    variance = sum((it - mu)**2 for it in seq) / n
    return torch.sqrt(variance)


def std_error(seq):
    n = len(seq)
    return std(seq) / torch.sqrt(torch.tensor(n).float())


@curry
def quantile(seq, q):
    arr = torch.stack(seq).detach().numpy()
    ans = np.quantile(arr, q, axis=0)
    return torch.tensor(ans)


def plot_with_dashes(args, qt_level=15, sli_level=18):
    """Chris wanted to see dashes where inputs are ignored"""
    axs = plot(args)

    p = args[1]

    # QT input
    for ax in axs[:, 0]:
        ax.axvline(p[qt_level], linestyle='--', c='k')

    # SLI input
    for ax in axs[:, 1]:
        ax.axvline(p[sli_level], linestyle='--', c='k')


def get_data(
        model_path="../../models/265/5.pkl",
        y_index=32, ):
    # open model
    model = torch.load(model_path)
    model.eval()

    # get data
    ds = xr.open_dataset(ngaqua_climate_path)
    ds = ds.expand_dims('x', axis=-1)

    location = ds.isel(y=slice(y_index, y_index + 1))
    M = saliency_map_one_location(model, location)
    return M, ds.p.values.ravel()


def main():
    import sys
    unstable_path, stable_path = sys.argv[1:]
    nn_all_path = "nn/NNAll/5.pkl"
    nn_lower_path = "nn/NNLower/5.pkl"
    
    y_index = 32
    
    axs = plot(get_data(nn_all_path, y_index))
    plt.savefig(unstable_path)

    args = get_data(nn_lower_path, y_index)
    plot_with_dashes(args)
    plt.savefig(stable_path)


if __name__ == '__main__':
    main()
