from functools import lru_cache
import numpy as np
from collections import Counter
import os
from os.path import isfile, join
import torch
from uwnet.stochastic_parameterization.graph_utils import draw_barplot_multi


dir_ = '/Users/stewart/Desktop/stochastic_param_data/stochastic_python_output'
eta_key = 'tendency_of_stochastic_state_due_to_neural_network'


def get_file_filter(file_prefix, file_suffix):
    if file_suffix and file_prefix:
        return lambda f: f.startswith(file_prefix) and f.endswith(
            file_suffix)
    elif file_suffix and not file_prefix:
        return lambda f: f.endswith(file_suffix)
    elif file_prefix and not file_suffix:
        return lambda f: f.startswith(file_prefix)
    return lambda f: True


def get_files_in_directory(directory, file_suffix='', file_prefix=''):
    filter_ = get_file_filter(file_prefix, file_suffix)
    return [
        f for f in os.listdir(directory) if
        (isfile(join(directory, f)) and filter_(f) and (f != '.DS_Store'))
    ]


@lru_cache()
def load_eta_data_files():
    etas = []
    files = get_files_in_directory(dir_, file_suffix='.pt')
    for filename in sorted(files):
        data = torch.load(dir_ + '/' + filename)
        if eta_key in data:
            etas.append(data[eta_key])
    return np.stack(etas)


def plot_tropics_eta_distribution_over_time():
    etas = load_eta_data_files()
    possible_etas = np.unique(etas)
    etas_tropics = etas[:, 28:36, :]
    etas_tropics_daily = etas_tropics[np.arange(len(etas_tropics)) % 8 == 0]
    distributions = []
    for row in etas_tropics_daily:
        counts = Counter(row.ravel())
        distributions.append(
            [counts.get(eta, 0) for eta in possible_etas]
        )
    legend_labels = [f'Eta = {eta}' for eta in possible_etas]
    draw_barplot_multi(
        np.array(distributions).T,
        list(range(len(distributions))),
        legend_labels=legend_labels,
        title='Distribution of Etas over Time')


if __name__ == '__main__':
    plot_tropics_eta_distribution_over_time()
