# Standard Library
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Thirdparty
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score

# uwnet
from uwnet.tensordict import TensorDict
from uwnet.stochastic_parameterization.residual_stochastic_state_model import (  # noqa
    StochasticStateModel,
)
from uwnet.stochastic_parameterization.utils import (
    get_dataset,
)
from uwnet.stochastic_parameterization.graph_utils import (
    draw_histogram,
    draw_barplot_multi,
    loghist,
)

dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
ds_location = dir_ + 'training.nc'
model = StochasticStateModel(
    ds_location=ds_location,
    base_model_location=dir_ + 'full_model/1.pkl'
)
model.train()
base_model = torch.load(dir_ + 'full_model/1.pkl')


def r2_score_(pred, truth, weights, dims=(0, 2, 3)):
    """ R2 score for xarray objects
    """
    mu = truth.mean(dims)
    sum_squares_error = ((truth - pred)**2).mean(dims)
    truth_normalized = truth.copy()
    for i in range(len(mu)):
        truth_normalized[:, i, :, :] -= mu[i]

    sum_squares = (truth_normalized**2).mean(dims)

    r2s = 1 - sum_squares_error / sum_squares
    return np.average(r2s, weights=weights)


def get_true_nn_forcing(time_, ds):
    start = ds.isel(time=time_)
    stop = ds.isel(time=time_ + 1)
    true_nn_forcing = {}
    for key in ['QT', 'SLI']:
        forcing = (start[f'F{key}'] + stop[f'F{key}']) / 2
        true_nn_forcing[key] = (stop[key] - start[key] - (
            10800 * forcing)).values / .125
    return true_nn_forcing


def predict_for_time(time_, ds, model=model, true_etas=True):
    ds_filtered = ds.isel(time=time_)
    to_predict = {}
    for key_ in ds.data_vars:
        if len(ds_filtered[key_].values.shape) == 2:
            val = ds_filtered[key_].values[np.newaxis, :, :]
        else:
            val = ds_filtered[key_].values.astype(np.float64)
        to_predict[key_] = torch.from_numpy(val)
    if hasattr(model, 'eta_transitioner') and true_etas:
        pred = model(
            TensorDict(to_predict),
            eta=ds_filtered.eta.values)
    else:
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
    ds = get_dataset(
        ds_location=ds_location,
        base_model_location=dir_ + 'full_model/1.pkl')
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


def simulate_eta(ds):
    etas = []
    for time in range(len(ds.time)):
        etas.append(model.eta)
        input_data = {}
        for predictor in model.eta_transitioner.predictors:
            input_data[predictor] = torch.from_numpy(
                ds.isel(time=time)[predictor].values)
        model.update_eta(TensorDict(input_data))
    return np.stack(etas)


def plot_true_eta_vs_simulated_eta(ds=None):
    if not ds:
        ds = get_dataset(
            ds_location=ds_location,
            base_model_location=dir_ + 'full_model/1.pkl',
            t_start=50,
            t_stop=75)
    simulated_eta = simulate_eta(ds)
    true_eta = ds.eta.values
    for eta in range(ds.eta.values.min(), ds.eta.values.max() + 1):
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


def trim_extreme_values(array):
    array = np.array(array)
    return array[
        (array > np.percentile(array, 3)) &
        (array < np.percentile(array, 97))
    ]


def get_column_moistening_and_heating_comparisons(true_etas=True):
    ds = get_dataset(
            ds_location=ds_location,
            base_model_location=dir_ + 'full_model/1.pkl',
            t_start=50,
            t_stop=75)
    layer_mass_sum = ds.layer_mass.values.sum()
    qts_pred = []
    qts_true = []
    qts_pred_base = []
    slis_pred = []
    slis_true = []
    slis_pred_base = []
    for time in range(24):
        pred = predict_for_time(time, ds, true_etas=true_etas)
        pred_base = predict_for_time(time, ds, base_model)
        true = get_true_nn_forcing(time, ds)
        qts_pred.extend(
            pred['QT'].T.dot(ds.layer_mass.values).ravel() / layer_mass_sum)
        qts_true.extend(
            true['QT'].T.dot(ds.layer_mass.values).ravel() / layer_mass_sum)
        qts_pred_base.extend(
            pred_base['QT'].T.dot(ds.layer_mass.values).ravel() /
            layer_mass_sum)
        slis_pred.extend(
            pred['SLI'].T.dot(ds.layer_mass.values).ravel() / layer_mass_sum)
        slis_true.extend(
            true['SLI'].T.dot(ds.layer_mass.values).ravel() / layer_mass_sum)
        slis_pred_base.extend(
            pred_base['SLI'].T.dot(ds.layer_mass.values).ravel() /
            layer_mass_sum)
    qts_true = np.array(qts_true)
    qts_pred = np.array(qts_pred)
    qts_pred_base = np.array(qts_pred_base)
    slis_true = np.array(slis_true)
    slis_pred = np.array(slis_pred)
    slis_pred_base = np.array(slis_pred_base)
    return (
        qts_true,
        qts_pred,
        qts_pred_base,
        slis_true,
        slis_pred,
        slis_pred_base,
    )


def evaluate_stochasticity_of_model(n_simulations=20):
    ds = get_dataset(
        ds_location=ds_location,
        base_model_location=dir_ + 'full_model/1.pkl',
        t_start=50,
        t_stop=75)
    max_probs = []
    etas = []
    for time in range(n_simulations):
        etas.append(model.eta)
        input_data = {}
        for predictor in model.eta_transitioner.predictors:
            input_data[predictor] = torch.from_numpy(
                ds.isel(time=time)[predictor].values)
        input_to_transitioner_model = \
            model.eta_transitioner.get_input_array_from_state(
                model.eta, TensorDict(input_data))
        transition_probs = model.eta_transitioner.model.predict_proba(
            input_to_transitioner_model)
        max_probs.extend(transition_probs[
            range(len(transition_probs)), transition_probs.argmax(axis=1)
        ].tolist())
        model.update_eta(TensorDict(input_data))
    max_probs = np.array(max_probs)
    print(f'Median max transition prob: {np.median(max_probs)}')
    print(f'Mean max transition prob: {max_probs.mean()}')
    print(f'Variance of max transition probs: {max_probs.var()}')
    draw_histogram(
        max_probs, pdf=True, title='PDF of max transition probability')


def compare_true_to_simulated_q1_q2_distributions(true_etas=True):
    (
        qts_true,
        qts_pred,
        qts_pred_base,
        slis_true,
        slis_pred,
        slis_pred_base,
    ) = get_column_moistening_and_heating_comparisons(true_etas=true_etas)
    true_qt_variance = qts_true.var()
    print(f'True QT Variance: {true_qt_variance}')
    pred_qt_variance = qts_pred.var()
    print(f'Stochastic QT Variance: {pred_qt_variance}')
    pred_base_qt_variance = qts_pred_base.var()
    print(f'Base Model QT Variance: {pred_base_qt_variance}')
    true_sli_variance = slis_true.var()
    print(f'True sli Variance: {true_sli_variance}')
    pred_sli_variance = slis_pred.var()
    print(f'Stochastic sli Variance: {pred_sli_variance}')
    pred_base_sli_variance = slis_pred_base.var()
    print(f'Base Model sli Variance: {pred_base_sli_variance}')
    print(f'\n\nSLI R2 Stochastic Model:',
          f' {r2_score(slis_true, slis_pred)}')
    print(f'SLI R2 Single Model Model:',
          f' {r2_score(slis_true, slis_pred_base)}')
    print(f'QT R2 Stochastic Model:',
          f' {r2_score(qts_true, qts_pred)}')
    print(f'QT R2 Single Model Model:',
          f' {r2_score(qts_true, qts_pred_base)}')

    qt_true_vs_base_ks = ks_2samp(qts_true, qts_pred_base)
    print('\n\nKS Divergence test: QT true vs single model: {}'.format(
        qt_true_vs_base_ks))
    qt_true_vs_stochastic_ks = ks_2samp(qts_true, qts_pred)
    print('KS Divergence test: QT true vs stochastic model: {}'.format(
        qt_true_vs_stochastic_ks))

    sli_true_vs_base_ks = ks_2samp(slis_true, slis_pred_base)
    print('KS Divergence test: SLI true vs single model: {}'.format(
        sli_true_vs_base_ks))
    sli_true_vs_stochastic_ks = ks_2samp(slis_true, slis_pred)
    print('KS Divergence test: SLI true vs stochastic model: {}'.format(
        sli_true_vs_stochastic_ks))
    fig, ax = plt.subplots()
    loghist(
        slis_true,
        ax=ax,
        upper_percentile=99.9,
        lower_percentile=0.01,
        label='True',
        gaussian_comparison=False
    )
    loghist(
        slis_pred,
        ax=ax,
        upper_percentile=99.9,
        lower_percentile=0.01,
        label='Stochastic Model',
        gaussian_comparison=False
    )
    loghist(
        slis_pred_base,
        ax=ax,
        upper_percentile=99.9,
        lower_percentile=0.01,
        label='Single Model',
        gaussian_comparison=False
    )
    plt.legend()
    if true_etas:
        title = 'Log Histogram for NN SLI Forcing, with True Eta Transitions'
    else:
        title = 'Log Histogram for NN SLI Forcing, with Simulated Eta Transitions'  # noqa
    plt.title(title)
    plt.show()

    fig, ax = plt.subplots()
    loghist(
        qts_true,
        ax=ax,
        lower_percentile=.03,
        label='True',
        gaussian_comparison=False
    )
    loghist(
        qts_pred,
        ax=ax,
        lower_percentile=.01,
        label='Stochastic Model',
        gaussian_comparison=False
    )
    loghist(
        qts_pred_base,
        ax=ax,
        lower_percentile=.01,
        label='Single Model',
        gaussian_comparison=False
    )
    plt.legend()
    plt.title('Log Histogram for NN QT Forcing, with True Eta Transitions')
    plt.show()


if __name__ == '__main__':
    # evaluate_stochasticity_of_model()
    # plot_true_eta_vs_simulated_eta()
    compare_true_to_simulated_q1_q2_distributions(False)
