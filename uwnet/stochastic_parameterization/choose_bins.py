from copy import copy
from uwnet.stochastic_parameterization.utils import (
    get_dataset,
)
import numpy as np
dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
base_model_location = dir_ + 'full_model/1.pkl'
ds_location = dir_ + 'training.nc'
data = get_dataset(
    t_start=400,
    t_stop=500,
    base_model_location=base_model_location,
    ds_location=ds_location,
    binning_quantiles=(1,)
).column_integrated_qt_residuals.values.ravel()

dl = 0.0001
max_iter = 1000
learning_rate = 0.001
min_distance_between_quantiles = 0.0001


def get_bin_membership(binning_quantiles):
    bin_membership = []
    prev_quantile = 0
    for quantile in binning_quantiles:
        bin_membership.append(data[
            (data >= np.quantile(data, prev_quantile)) &
            (data < np.quantile(data, quantile))
        ])
        prev_quantile = quantile
    return bin_membership


def loss(binning_quantiles):
    numerator = 0
    denominator = 0
    for bin_ in get_bin_membership(binning_quantiles):
        numerator += len(bin_) * bin_.var()
        denominator += len(bin_)
    loss_ = numerator / denominator
    return loss_


def estimate_gradient(binning_quantiles):
    base_loss = loss(binning_quantiles)
    gradient = np.zeros(len(binning_quantiles))
    for idx, quantile in enumerate(binning_quantiles[:-1]):
        max_ = binning_quantiles[idx + 1] - min_distance_between_quantiles
        if idx == 0:
            min_ = min_distance_between_quantiles
        else:
            min_ = binning_quantiles[idx - 1] + min_distance_between_quantiles
        binning_quantiles_dl = list(copy(binning_quantiles))
        binning_quantiles_dl[idx] = max(min(quantile + dl, max_), min_)
        grad_for_quantile = (
            loss(binning_quantiles_dl) - base_loss) / dl
        gradient[idx] = grad_for_quantile
    return gradient


def update_binning_quantiles(binning_quantiles):
    gradient = estimate_gradient(binning_quantiles)
    new_binning_quantiles = copy(binning_quantiles)
    for idx, grad in enumerate(gradient[:-1]):
        current_quantile = binning_quantiles[idx]
        max_ = binning_quantiles[idx + 1] - min_distance_between_quantiles
        if idx == 0:
            min_ = min_distance_between_quantiles
        else:
            min_ = new_binning_quantiles[
                idx - 1] + min_distance_between_quantiles
        new_quantile = max(min(
            current_quantile - (grad * learning_rate), max_), min_)
        new_binning_quantiles[idx] = new_quantile
    return new_binning_quantiles


def optimize_loss(starting_binning_quantiles):
    current_loss = loss(starting_binning_quantiles)
    binning_quantiles = starting_binning_quantiles
    best_loss = float('inf')
    best_binning_quantile = copy(starting_binning_quantiles)
    idx = 0
    while (idx < max_iter):
        binning_quantiles = update_binning_quantiles(binning_quantiles)
        current_loss = loss(binning_quantiles)
        print(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_binning_quantile = copy(binning_quantiles)
        idx += 1
        if not idx % 20:
            print(best_binning_quantile)
    return best_binning_quantile


if __name__ == '__main__':
    optimize_loss(
        np.array([0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1]))
