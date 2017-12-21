"""Module for computing and plotting linear response functions
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.externals import joblib
from .util import compute_dp


def get_lrf(lm_data):
    lrf, in_idx, out_idx = lm_data['mat'], lm_data['features']['in'], lm_data['features']['out']
    return pd.DataFrame(lrf, index=in_idx, columns=out_idx)


def plot_lrf(lrf,
             p,
             width_ratios=[1, 1, .3, .3],
             figsize=(10, 5),
             image_kwargs={}):
    """Plot linear response function"""
    p = np.asarray(p)

    # read input and output variables from lrf
    input_vars = lrf.index.levels[0]
    output_vars = lrf.columns.levels[0]
    width_ratios = width_ratios[:len(input_vars)]

    ni, no = len(input_vars), len(output_vars)
    print("Making figure with", ni, "by", no, "panes")

    fig, axs = plt.subplots(no, ni, figsize=figsize)

    grid = gridspec.GridSpec(2, len(input_vars), width_ratios=width_ratios)

    for j, input_var in enumerate(input_vars):
        for i, output_var in enumerate(output_vars):
            ax = plt.subplot(grid[i, j])
            lrf_pane = np.asarray(lrf.loc[input_var][output_var])
            #             print(lrf_pane.shape)
            if lrf_pane.shape[0] == 1:
                ax.plot(lrf_pane.ravel(), p)
                ax.invert_yaxis()
            else:
                # compute pressure difference
                dp = compute_dp(p)
                im = ax.pcolormesh(p, p, (lrf_pane).T / dp, **image_kwargs)
                # make colorbar
                cbar = plt.colorbar(im, orientation='vertical', pad=.02)
                # invert y axis for pressure coordinates
                ax.invert_yaxis()
                ax.invert_xaxis()

            # turn off axes ylabels
            if j > 0:
                ax.set_yticks([])
            if i < no - 1:
                ax.set_xticks([])

            # Add variable names
            if j == 0:
                ax.set_ylabel(output_var)

            if i == 0:
                ax.set_title(input_var)

    return fig, axs


def plot_lrf_from_file(lm_file, stat_file, label=""):
    mca_regression = joblib.load(lm_file)


    lrf = get_lrf(mca_regression)
    p = xr.open_dataset(stat_file).p
    plot_lrf(lrf, p, input_vars=['qt', 'sl'], width_ratios=[1,1],
             output_vars=['Q1c', 'Q2'])

    score = mca_regression['test_score']

    plt.suptitle('{}; R2 Score: {:.2f}'.format(label, score))
