# Standard Library
import random
from collections import defaultdict

# Thirdparty
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

# uwnet
from uwnet.tensordict import TensorDict
from stochastic_parameterization.stochastic_state_model import (  # noqa
    StochasticStateModel,
)
from stochastic_parameterization.utils import get_dataset
from stochastic_parameterization.graph_utils import (
    draw_histogram,
    draw_barplot_multi,
)

model_dir = '/Users/stewart/projects/uwnet/stochastic_parameterization'
model_location = model_dir + '/stochastic_model.pkl'
binning_method = 'precip'
# binning_method = 'q2_ratio'
base_model_location = model_dir + '/full_model/1.pkl'
model = torch.load(model_location)


def get_true_nn_forcing(time_, ds):
    start = ds.isel(time=time_)
    stop = ds.isel(time=time_ + 1)
    true_nn_forcing = {}
    for key in ['QT', 'SLI']:
        forcing = (start[f'F{key}'] + stop[f'F{key}']) / 2
        true_nn_forcing[key] = (stop[key] - start[key] - (
            10800 * forcing)).values / .125
    return true_nn_forcing


def predict_for_time(time_, ds):
    ds_filtered = ds.isel(time=time_)
    to_predict = {}
    for key_ in ds.data_vars:
        if len(ds_filtered[key_].values.shape) == 2:
            val = ds_filtered[key_].values[np.newaxis, :, :]
        else:
            val = ds_filtered[key_].values.astype(np.float64)
        to_predict[key_] = torch.from_numpy(val)
    pred = model(TensorDict(to_predict))
    return {key: pred[key].detach().numpy() for key in pred}


def get_layer_mass_averaged_residuals_for_time(time_, ds, layer_mass_sum):
    predictions = predict_for_time(time_, ds)
    true_values = get_true_nn_forcing(time_, ds)
    residuals = {}
    for key, val in predictions.items():
        residuals[key] = (val - true_values[key]).T.dot(
            ds.layer_mass.values) / layer_mass_sum
    return residuals


def plot_residuals_by_eta():
    ds = get_dataset()
    layer_mass_sum = ds.layer_mass.values.sum()
    qt_residuals_by_eta = defaultdict(list)
    sli_residuals_by_eta = defaultdict(list)
    for _ in range(10):
        time_ = random.choice(range(len(ds.time)))
        residuals = get_layer_mass_averaged_residuals_for_time(
            time_, ds, layer_mass_sum)
        for eta_ in model.possible_etas:
            indices = np.argwhere(model.eta == eta_)
            qt_residuals_by_eta[eta_].extend(
                residuals['QT'][indices[:, 1], indices[:, 0]].tolist())
            sli_residuals_by_eta[eta_].extend(
                residuals['SLI'][indices[:, 1], indices[:, 0]].tolist())
    for eta, residuals in qt_residuals_by_eta.items():
        draw_histogram(residuals, title=f'QT residuals for eta={eta}')
    for eta, residuals in sli_residuals_by_eta.items():
        draw_histogram(residuals, title=f'SLI residuals for eta={eta}')


def simulate_eta():
    etas = []
    for _ in range(640):
        etas.append(model.eta)
        model.update_eta()
    return np.stack(etas)


def plot_true_eta_vs_simulated_eta(ds=None):
    if not ds:
        ds = get_dataset()
    simulated_eta = simulate_eta()
    true_eta = ds.eta.values
    for eta in range(ds.eta.values.min(), ds.eta.values.max()):
        true_y_distribution = pd.DataFrame(
            np.argwhere(true_eta == eta), columns=['time', 'y', 'x']
        ).y.value_counts().sort_index().tolist()
        simulated_y_distribution = pd.DataFrame(
            np.argwhere(simulated_eta == eta), columns=['time', 'y', 'x']
        ).y.value_counts().sort_index().tolist()
        draw_barplot_multi(
            [true_y_distribution, simulated_y_distribution],
            [''] * len(ds.y),
            title=f'Simulated vs True Y Distribution of eta = {eta}',
            legend_labels=['True', 'Simulated']
        )


def compare_true_to_simulated_q1_q2_distributions():
    ds = get_dataset()
    layer_mass_sum = ds.layer_mass.values.sum()
    times = random.sample(range(len(ds.time)), 10)
    qts_pred = []
    qts_true = []
    slis_pred = []
    slis_true = []
    for time in times:
        pred = predict_for_time(time, ds)
        true = get_true_nn_forcing(time, ds)
        qts_pred.extend(
            pred['QT'].T.dot(ds.layer_mass.values).ravel() / layer_mass_sum)
        qts_true.extend(
            true['QT'].T.dot(ds.layer_mass.values).ravel() / layer_mass_sum)
        slis_pred.extend(
            pred['SLI'].T.dot(ds.layer_mass.values).ravel() / layer_mass_sum)
        slis_true.extend(
            true['SLI'].T.dot(ds.layer_mass.values).ravel() / layer_mass_sum)
    print(f'SLI R2: {r2_score(slis_pred, slis_true)}')
    print(f'QT R2: {r2_score(qts_pred, qts_true)}')
