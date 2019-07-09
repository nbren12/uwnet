import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data import open_data
from uwnet.thermo import *


def tropical_data():
    # load model and datya
    ds = open_data('training').sel(time=slice(120,140))
    p = open_data('pressure')

    # make necessary computations
    lat = ngaqua_y_to_lat(ds.y)
    # compute LTS and Mid Trop moisture
    ds['lts'] = lower_tropospheric_stability(ds.TABS, p, ds.SST, ds.Ps)
    ds['path'] = midtropospheric_moisture(ds.QV, p, bottom=850, top=600)

    # select the tropics
    tropics = ds.isel(y=(np.abs(lat)<  11))

    return tropics


def compute_nn(model, averages, batch_dims):
    bins_key = batch_dims[0]
    avgs_expanded = averages.rename({bins_key: 'time'}).expand_dims(['x', 'y'], [-1, -2])
    return model.call_with_xr(avgs_expanded).rename({'time': bins_key}).squeeze()


def groupby_and_compute_nn(tropics, model, key, bins):
    bins_key = key + '_bins'
    averages = (tropics
     .stack(gridcell=['x', 'y', 'time'])
     .groupby_bins(key, bins=bins)
     .mean('gridcell'))

    output = compute_nn(model, averages, batch_dims=[bins_key])
    for key in output:
        NNkey = 'NN' + key
        averages[NNkey] = output[key]
        
    return averages


def plot_line_cmap(arr, lower_val=4, key='path_bins'):
    val = [bin.mid for bin in arr[key].values]
    for it, arr in arr.groupby(key):
        label = it.mid
        arr.plot(y='z', hue=key, color=plt.cm.inferno((it.mid + lower_val)/(25+lower_val)), label=label)
    plt.legend()

    
if __name__ == '__main__':
    # output paths
    lts_plot_path = "sensitivity_to_lts.pdf"
    moisture_plot_path = "sensitivity_to_moist.pdf"
    model_path = "../../nn/NNLowerDecayLR/20.pkl"

    # bins for moisture and lower tropospheric stability
    moisture_bins = np.r_[:28:2.5]
    lts_bins = np.r_[2:20:1]

    # load data
    tropics = tropical_data()
    model = torch.load(model_path)
    
    
    # Lower tropospheric stability plots
    output_lts_binned = groupby_and_compute_nn(tropics, model=model, key='lts', bins=lts_bins)
    plt.subplot(121)
    plot_line_cmap(output_lts_binned.NNQT, key='lts_bins')
    plt.subplot(122)
    plot_line_cmap(output_lts_binned.NNSLI, key='lts_bins')
    plt.savefig(lts_plot_path)

    # Mid-tropospheric moisture plots
    plt.figure()
    output = groupby_and_compute_nn(tropics, model, 'path', moisture_bins)
    plt.subplot(121)
    plot_line_cmap(output.NNQT)
    plt.subplot(122)
    plot_line_cmap(output.NNSLI)
    plt.savefig(moisture_plot_path)