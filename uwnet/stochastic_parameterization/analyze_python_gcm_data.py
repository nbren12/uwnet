from functools import lru_cache
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from uwnet.thermo import integrate_q2, compute_apparent_source
import os
from os.path import isfile, join
import torch
from uwnet.stochastic_parameterization.graph_utils import draw_barplot_multi
from uwnet.stochastic_parameterization.stochastic_state_model import (
    StochasticStateModel
)
from uwnet.stochastic_parameterization.utils import get_dataset


dir_ = '/Users/stewart/Desktop/stochastic_param_data/stochastic_python_output'
eta_key = 'tendency_of_stochastic_state_due_to_neural_network'
ds_dir = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
name_dict = {
    "liquid_ice_static_energy": "SLI",
    "x_wind": "U",
    "y_wind": "V",
    "upward_air_velocity": "W",
    "total_water_mixing_ratio": "QT",
    "air_temperature": "TABS",
    "latitude": "lat",
    "longitude": "lon",
    "sea_surface_temperature": "SST",
    "surface_air_pressure": "p0",
    "toa_incoming_shortwave_flux": "SOLIN",
    "surface_upward_sensible_heat_flux": "SHF",
    "surface_upward_latent_heat_flux": "LHF",
}


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


def load_files():
    files = get_files_in_directory(dir_, file_suffix='.pt')
    data = defaultdict(list)
    for filename in sorted(files):
        data_in_file = torch.load(dir_ + '/' + filename)
        for var, data_for_var in data_in_file.items():
            if var in name_dict:
                data[name_dict[var]].append(data_for_var)
            else:
                data[var].append(data_for_var)
    return {
        var: np.stack(data_for_var)
        for var, data_for_var in data.items()
    }


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
        title='Distribution of Etas by Day in Tropics')


def plot_nn_output_vs_true():
    no_param_run = load_files()
    ds_no_param = get_dataset(
        ds_location=ds_dir + "training.nc",  # noqa
        set_eta=False,
        t_start=0,
        t_stop=len(no_param_run))
    for key, vals in no_param_run.items():
        try:
            ds_no_param[key].values = vals
        except:  # noqa
            continue
    model = StochasticStateModel(
        t_start=50,
        t_stop=75,
        ds_location=ds_dir + "training.nc",
        base_model_location=ds_dir + 'full_model/1.pkl')
    model.train()
    ds_true = get_dataset(
        ds_location=ds_dir + "training.nc",  # noqa
        set_eta=False,
        t_start=0,
        t_stop=10)
    predicted = -integrate_q2(
        model.predict(ds_no_param).QT, ds_true.layer_mass).isel(time=-2)
    print(f'Predicted Mean Squared prediction: {(predicted ** 2).mean()}')
    predicted.plot()
    plt.title('Predicted on No Parameterization Data')
    plt.show()
    true = -integrate_q2(compute_apparent_source(
        ds_true.QT, ds_true.FQT * 86400),
        ds_true.layer_mass).isel(time=-2)
    print(f'True Mean Squared prediction: {(true ** 2).mean()}')
    true.plot()
    plt.title('True ')
    plt.show()


if __name__ == '__main__':
    plot_nn_output_vs_true()
