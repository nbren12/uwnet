# Standard Library
import random
from collections import defaultdict

# Thirdparty
import numpy as np
import pandas as pd
import torch

# Uwnet
from uwnet.tensordict import TensorDict
from uwnet.train import get_xarray_dataset
from stochastic_parameterization.stochastic_state_model import (  # noqa
    StochasticStateModel,
)
from stochastic_parameterization.graph_utils import (
    draw_histogram,
    draw_barplot_multi,
)

model_location = '/Users/stewart/projects/uwnet/stochastic_parameterization/stochastic_model.pkl'  # noqa

model = torch.load(model_location)
ds = get_xarray_dataset(
    "/Users/stewart/projects/uwnet/data/processed/training.nc",
    model.precip_quantiles
)


def get_true_nn_forcing(time_):
    start = ds.isel(time=time_)
    stop = ds.isel(time=time_ + 1)
    true_nn_forcing = {}
    for key in ['QT', 'SLI']:
        true_nn_forcing[key] = (stop['QT'] - start['QT'] - (
            10800 * start['FQT'])).values
    return true_nn_forcing


def predict_for_time(time_):
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


def get_layer_mass_averaged_residuals_for_time(time_):
    predictions = predict_for_time(time_)
    true_values = get_true_nn_forcing(time_)
    residuals = {}
    for key, val in predictions.items():
        residuals[key] = (val - true_values[key]).T.dot(ds.layer_mass.values)
    return residuals


def plot_residuals_by_eta():
    qt_residuals_by_eta = defaultdict(list)
    sli_residuals_by_eta = defaultdict(list)
    for _ in range(10):
        time_ = random.choice(range(len(ds.time)))
        residuals = get_layer_mass_averaged_residuals_for_time(time_)
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


def plot_true_eta_vs_simulated_eta():
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
    times = random.sample(range(len(ds.time)), 10)
    qts_pred = []
    qts_true = []
    slis_pred = []
    slis_true = []
    for time in times:
        pred = predict_for_time(time)
        true = get_true_nn_forcing(time)
        qts_pred.extend(pred['QT'].T.dot(ds.layer_mass.values).ravel())
        qts_true.extend(true['QT'].T.dot(ds.layer_mass.values).ravel())
        slis_pred.extend(pred['SLI'].T.dot(ds.layer_mass.values).ravel())
        slis_true.extend(true['SLI'].T.dot(ds.layer_mass.values).ravel())
