"""Plots of the training process for the neural network"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from toolz import merge


def open_json(path, **kwargs):
    d = json.load(open(path))

    metadata = merge(d['args'], kwargs)
    metadata['path'] = path
    metadata['nhidden'] = metadata['nhidden'][0]

    df = pd.DataFrame(d['training']).assign(**metadata)
    return df


def open_model(path, **kwargs):
    seeds = glob.glob(path + "/*.json")
    for seed in seeds:
        i = os.path.splitext(os.path.basename(seed))[0]
        yield open_json(seed, seed=i, **kwargs)


def _open_jsons(path):
    models = glob.glob(path + "/model.*")
    models = filter(os.path.isdir, models)
    for model in models:
        m = re.search(r"model\.(.+)", os.path.basename(model))
        yield from open_model(model, model=m.group(1))


def open_jsons(path):
    return pd.concat(_open_jsons(path), axis=0)


def get_plotting_data(path):
    df = open_jsons(path)

    # fill in Na to training loss when batch = 0
    df.loc[df.batch == 0, 'train_loss'] = np.NaN
    # only keep a subet of the variables
    variables = [
        'nhidden', 'window_size', 'test_loss', 'train_loss', 'epoch', 'batch',
        'seed'
    ]
    df = df[variables]

    # Get vary num hidden experiment
    nhid = df[df.window_size == 10]
    vt = df[df.nhidden == 128]

    # remove outliers
    nhid = nhid[~((nhid.seed == '6') & (nhid.nhidden == 256))]
    vt = vt[vt.window_size != 2]

    return vt, nhid


def plot_parameter_sensitivity(data):
    vt, nhid = data
    fig, (axn, axt) = plt.subplots(
        1,
        2,
        figsize=(6, 6 / 1.62),
        sharey=True,
        gridspec_kw=dict(width_ratios=(.67, .33), wspace=0))

    kws = dict(capsize=.2, join=True)

    sns.pointplot(
        x="nhidden", y="test_loss", data=nhid[nhid.epoch > 2], ax=axn, **kws)
    sns.pointplot(
        x="window_size", y="test_loss", data=vt[vt.epoch > 2], ax=axt, **kws)

    plt.ylim([120, 155])
    axt.set_ylabel('')
    axn.set_ylabel('Error')

    axn.set_xlabel("Hidden Nodes")
    axt.set_xlabel("Window Size")
    fig.suptitle("Test error for last 4 epochs")


def plot_epochs_vs_loss(data):
    vt, nhid = data

    def plot_train_test(val, axtrain, axtest, **kwargs):
        stats = val.groupby('epoch').median()
        stats.test_loss.plot(ax=axtest, **kwargs)
        stats.train_loss.plot(ax=axtrain, **kwargs)

    legend_box = (1.0, .7)
    fig, (axtrain, axtest) = plt.subplots(1, 2, figsize=(6, 3))

    kws = dict(marker='s')

    lines_nhid = []
    lines_vt = []

    alpha = .2
    for n, val in nhid.groupby('nhidden'):
        plot_train_test(
            val, axtrain, axtest, color='b', alpha=alpha, label=n, **kws)
        lines_nhid.append(n)
        alpha += .2

    leg1 = plt.legend(
        axtest.get_lines(),
        lines_nhid,
        title='Hidden Nodes',
        loc="upper left",
        bbox_to_anchor=legend_box)

    alpha = .5
    for T, val in vt.groupby('window_size'):
        plot_train_test(
            val, axtrain, axtest, color='r', label=T, alpha=alpha, **kws)
        lines_vt.append(T)
        alpha += .5

    axtest.add_artist(leg1)

    leg2 = plt.legend(
        axtest.get_lines()[-2:],
        lines_vt,
        title='Window Size',
        loc="lower left",
        bbox_to_anchor=legend_box)

    axtest.set_ylim([120, 160])

    # labels
    axtrain.set_title('Training Loss')
    axtest.set_title('Median Test Error')
