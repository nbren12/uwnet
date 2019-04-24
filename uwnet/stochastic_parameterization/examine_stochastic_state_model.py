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
from uwnet.thermo import (
    compute_apparent_source,
    liquid_water_density,
    cp,
    sec_in_day,
)

dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
ds_location = dir_ + 'training.nc'
model = None
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
    ds_for_time = ds.isel(time=[time_, time_ + 1])
    qt_forcing = compute_apparent_source(
        ds_for_time.QT,
        ds_for_time.FQT * 86400).isel(time=0).values
    sli_forcing = compute_apparent_source(
        ds_for_time.SLI,
        ds_for_time.FSLI * 86400).isel(time=0).values
    return {'QT': qt_forcing, 'SLI': sli_forcing}


def predict_for_time(time_, ds, model=model, true_etas=True):
    ds_filtered = ds.isel(time=[time_])
    if hasattr(model, 'eta_transitioner') and true_etas:
        pred = model.predict(
            ds_filtered, eta=ds_filtered.isel(time=0).eta.values)
    else:
        pred = model.predict(ds_filtered)
    pred = pred.isel(time=0)
    return {key: pred[key].values for key in pred}


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


def get_column_moistening_and_heating_comparisons(
        model,
        true_etas=True,
        only_tropics=False,
        use_true_eta_start=True):
    ds = get_dataset(
        ds_location=ds_location,
        base_model_location=dir_ + 'full_model/1.pkl',
        t_start=50,
        t_stop=100)
    if use_true_eta_start:
        model.eta = ds.isel(time=0).eta.values
    qts_pred = []
    qts_true = []
    qts_pred_base = []
    slis_pred = []
    slis_true = []
    slis_pred_base = []
    layer_mass = ds.layer_mass.values
    for time in range(1, 48):
        pred = predict_for_time(time, ds, model=model, true_etas=true_etas)
        pred_base = predict_for_time(time, ds, base_model)
        true = get_true_nn_forcing(time, ds)
        if only_tropics:
            for forcing in [pred, pred_base, true]:
                for var in ['QT', 'SLI']:
                    forcing[var] = forcing[var][:, 28:36, :]
        qts_pred.extend(
            pred['QT'].T.dot(layer_mass).ravel() / liquid_water_density)
        qts_true.extend(
            true['QT'].T.dot(layer_mass).ravel() / liquid_water_density)
        qts_pred_base.extend(
            pred_base['QT'].T.dot(layer_mass).ravel() / liquid_water_density)
        slis_pred.extend(
            pred['SLI'].T.dot(layer_mass).ravel() * (cp / sec_in_day))
        slis_true.extend(
            true['SLI'].T.dot(layer_mass).ravel() * (cp / sec_in_day))
        slis_pred_base.extend(
            pred_base['SLI'].T.dot(layer_mass).ravel() * (cp / sec_in_day))
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


def evaluate_stochasticity_of_model(model, n_simulations=20):
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


def compare_true_to_simulated_q1_q2_distributions(
        model,
        true_etas=False,
        only_tropics=False,
        use_true_eta_start=True,
        title_prefix='',
        print_stats=True):
    (
        qts_true,
        qts_pred,
        qts_pred_base,
        slis_true,
        slis_pred,
        slis_pred_base,
    ) = get_column_moistening_and_heating_comparisons(
        model,
        true_etas=true_etas,
        only_tropics=only_tropics,
        use_true_eta_start=use_true_eta_start)
    true_qt_variance = qts_true.var()
    pred_qt_variance = qts_pred.var()
    pred_base_qt_variance = qts_pred_base.var()
    true_sli_variance = slis_true.var()
    pred_sli_variance = slis_pred.var()
    pred_base_sli_variance = slis_pred_base.var()
    if print_stats:
        print(f'True QT Variance: {true_qt_variance}')
        print(f'Stochastic QT Variance: {pred_qt_variance}')
        print(f'Base Model QT Variance: {pred_base_qt_variance}')
        print(f'True sli Variance: {true_sli_variance}')
        print(f'Stochastic sli Variance: {pred_sli_variance}')
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
    qt_true_vs_stochastic_ks = ks_2samp(qts_true, qts_pred)

    sli_true_vs_base_ks = ks_2samp(slis_true, slis_pred_base)
    sli_true_vs_stochastic_ks = ks_2samp(slis_true, slis_pred)

    if print_stats:
        print('\n\nKS Divergence test: QT true vs single model: {}'.format(
            qt_true_vs_base_ks))
        print('KS Divergence test: QT true vs stochastic model: {}'.format(
            qt_true_vs_stochastic_ks))
        print('KS Divergence test: SLI true vs single model: {}'.format(
            sli_true_vs_base_ks))
        print('KS Divergence test: SLI true vs stochastic model: {}'.format(
            sli_true_vs_stochastic_ks))
    fig, ax = plt.subplots()
    loghist(
        slis_true,
        ax=ax,
        upper_percentile=99.99,
        lower_percentile=0.01,
        label='True',
        gaussian_comparison=False
    )
    loghist(
        slis_pred,
        ax=ax,
        upper_percentile=99.99,
        lower_percentile=0.01,
        label='Stochastic Model',
        gaussian_comparison=False
    )
    loghist(
        slis_pred_base,
        ax=ax,
        upper_percentile=99.99,
        lower_percentile=0.01,
        label='Single Model',
        gaussian_comparison=False
    )
    plt.legend()
    if true_etas:
        title = 'Log Histogram for NN SLI Forcing, with True Eta Transitions'
    else:
        title = 'Log Histogram for NN SLI Forcing, with Simulated Eta Transitions'  # noqa
    plt.title(title_prefix + title)
    plt.show()

    fig, ax = plt.subplots()
    loghist(
        qts_true,
        ax=ax,
        lower_percentile=.01,
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
    if true_etas:
        title = 'Log Histogram for NN QT Forcing, with True Eta Transitions'
    else:
        title = 'Log Histogram for NN QT Forcing, with Simulated Eta Transitions'  # noqa
    plt.title(title_prefix + title)
    plt.show()


if __name__ == '__main__':
    model = StochasticStateModel(
        ds_location=ds_location,
        eta_coarsening=2,
        base_model_location=dir_ + 'full_model/1.pkl',
        verbose=True,
        markov_process=True,
        include_output_in_transition_model=True
    )
    model.train()
    # evaluate_stochasticity_of_model(model)
    # plot_true_eta_vs_simulated_eta()
    compare_true_to_simulated_q1_q2_distributions(
        model, true_etas=False, only_tropics=False)
