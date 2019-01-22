import torch
import xarray as xr
import numpy as np
from stochastic_parameterization.utils import get_dataset

percentiles = [.01486, .10, .30, .70, .90, .9851432, 1]
time_idx = 0


def load_dataset(data="data/processed/training.nc", only_tropics=False):
    dataset = xr.open_dataset(data)
    if only_tropics:
        dataset = dataset.isel(y=list(range(28, 36)))
    return dataset


def get_indices_for_quantile(
        dataset, precip_values, min_quantile, max_quantile):
    min_precip = dataset.Prec.quantile(min_quantile)
    max_precip = dataset.Prec.quantile(max_quantile)
    filtered_precip_ds = dataset.where(dataset.Prec >= min_precip).where(
        dataset.Prec < max_precip).Prec.values
    return np.argwhere(~np.isnan(filtered_precip_ds))


def get_indices_by_quantile_dict(dataset, precip_values, percentiles):
    min_quantile = 0
    indices_by_quantile_dict = {}
    for percentile in percentiles:
        indices_by_quantile_dict[percentile] = get_indices_for_quantile(
            dataset, precip_values, min_quantile, percentile
        )
        min_quantile = percentile
    return indices_by_quantile_dict


def count_transition_occurences(starting_indices, ending_indices):
    time_step_after_start = starting_indices.copy()
    time_step_after_start[:, time_idx] += 1
    set_a = set([tuple(coords) for coords in time_step_after_start])
    set_b = set([tuple(coords) for coords in ending_indices])
    return len(set_a.intersection(set_b))


def count_total_starting_occurences(dataset, indices_by_quantile):
    max_time_idx = len(dataset.time) - 1
    return (indices_by_quantile[:, 0] != max_time_idx).sum()


def get_transition_matrix(percentiles, only_tropics=False):
    dataset = load_dataset(only_tropics=only_tropics)
    dataset = dataset.isel(time=(list(range(20, 25))))
    precip_values = dataset.Prec.values.ravel()
    indices_by_quantile_dict = get_indices_by_quantile_dict(
        dataset, precip_values, percentiles)
    transition_matrix = []
    for idx, quantile_row in enumerate(percentiles):
        transition_row = []
        n_in_quantile = count_total_starting_occurences(
            dataset, indices_by_quantile_dict[quantile_row])
        for quantile_col in percentiles:
            transition_counts = count_transition_occurences(
                indices_by_quantile_dict[quantile_row],
                indices_by_quantile_dict[quantile_col])
            transition_probability = transition_counts / n_in_quantile
            transition_row.append(transition_probability)
        # ensure probabilities sum to 1 (they are never more than 10e-5 off)
        transition_row[idx] += (1 - sum(transition_row))
        transition_matrix.append(transition_row)
    return np.array(transition_matrix)


def get_q2_ratio_transition_matrix():
    dataset = get_dataset()
    possible_etas = set(dataset.eta.values.ravel())
