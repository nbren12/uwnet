import numpy as np
from functools import lru_cache
import torch
from uwnet.tensordict import TensorDict
import xarray as xr

model_dir = '/Users/stewart/projects/uwnet/stochastic_parameterization'
model_location = model_dir + '/stochastic_model.pkl'
binning_method = 'precip'
# binning_method = 'q2_ratio'
base_model_location = model_dir + '/full_model/1.pkl'


class BaseModel(object):

    def __init__(
            self,
            model_location,
            dataset,
            binning_quantiles=[0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1],
            time_step_days=.125):
        self.model = torch.load(model_location)
        self.ds = dataset
        self.time_step_days = time_step_days
        self.time_step_seconds = 86400 / time_step_days
        self.binning_quantiles = binning_quantiles

    def get_true_nn_forcing_idx(self, time_idx):
        start = self.ds.isel(time=time_idx)
        stop = self.ds.isel(time=time_idx + 1)
        true_nn_forcing = {}
        for key in ['QT', 'SLI']:
            forcing = (start[f'F{key}'].values + stop[f'F{key}'].values) / 2
            true_nn_forcing[key] = (stop[key].values - start[key].values - (
                self.time_step_seconds * forcing)) / self.time_step_days
        return true_nn_forcing

    def predict_for_time_idx(self, time_idx):
        ds_filtered = self.ds.isel(time=time_idx)
        to_predict = {}
        for key_ in self.ds.data_vars:
            if len(ds_filtered[key_].values.shape) == 2:
                val = ds_filtered[key_].values[np.newaxis, :, :]
            else:
                val = ds_filtered[key_].values.astype(np.float64)
            to_predict[key_] = torch.from_numpy(val)
        pred = self.model(TensorDict(to_predict))
        return {key: pred[key].detach().numpy() for key in pred}

    def get_qt_ratios(self):
        qt_ratios = np.ones_like(self.ds.Prec.values)
        time_indices = list(range(len(self.ds.time) - 1))
        for time_batch in np.array_split(time_indices, 10):
            q2_preds = np.ma.average(
                self.predict_for_time_idx(time_batch)['QT'],
                axis=1,
                weights=self.ds.layer_mass.values
            )
            q2_true = np.ma.average(
                self.get_true_nn_forcing_idx(time_batch)['QT'],
                axis=1,
                weights=self.ds.layer_mass.values
            )
            qt_ratios[time_batch, :, :] = q2_preds / q2_true
        return qt_ratios

    def get_bin_membership(self):
        qt_ratios = self.get_qt_ratios()
        bins = [
            np.quantile(qt_ratios, quantile)
            for quantile in self.binning_quantiles
        ]
        return np.digitize(qt_ratios, bins, right=True)


def insert_precipitation_bin_membership(dataset, binning_quantiles):
    if not binning_quantiles:
        return dataset
    bins = [
        dataset.Prec.quantile(quantile).values
        for quantile in binning_quantiles
    ]
    eta_ = dataset['Prec'].copy()
    eta_.values = np.digitize(dataset.Prec.values, bins, right=True)
    dataset['eta'] = eta_
    dataset['eta'].attrs['units'] = ''
    dataset['eta'].attrs['long_name'] = 'Stochastic State'
    return dataset


def insert_nn_output_precip_ratio_bin_membership(
        dataset,
        binning_quantiles,
        base_model_location):
    base_model = BaseModel(
        base_model_location,
        dataset
    )
    bin_membership = base_model.get_bin_membership()
    eta_ = dataset['Prec'].copy()
    eta_.values = bin_membership
    dataset['eta'] = eta_
    dataset['eta'].attrs['units'] = ''
    dataset['eta'].attrs['long_name'] = 'Stochastic State'
    return dataset


def get_xarray_dataset_with_eta(
        data, binning_quantiles, binning_method, base_model_location):
    try:
        dataset = xr.open_zarr(data)
    except ValueError:
        dataset = xr.open_dataset(data)

    if binning_method == 'q2_ratio':
        dataset = insert_nn_output_precip_ratio_bin_membership(
            dataset, binning_quantiles, base_model_location)
    elif binning_method == 'precip':
        dataset = insert_precipitation_bin_membership(
            dataset, binning_quantiles)
    try:
        return dataset.isel(step=0).drop('step').drop('p')
    except Exception:
        return dataset


@lru_cache()
def get_dataset(
        binning_quantiles=[0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1],
        ds_location="/Users/stewart/projects/uwnet/data/processed/training.nc",
        binning_method='q2_ratio',
        base_model_location=base_model_location):
    return get_xarray_dataset_with_eta(
        "/Users/stewart/projects/uwnet/data/processed/training.nc",
        binning_quantiles,
        binning_method,
        base_model_location
    )


def count_transition_occurences(starting_indices, ending_indices):
    time_step_after_start = starting_indices.copy()
    time_step_after_start[:, 0] += 1
    set_a = set([tuple(coords) for coords in time_step_after_start])
    set_b = set([tuple(coords) for coords in ending_indices])
    return len(set_a.intersection(set_b))


def count_total_starting_occurences(dataset, indices_by_eta):
    max_time_idx = len(dataset.time) - 1
    return (indices_by_eta[:, 0] != max_time_idx).sum()


def get_q2_ratio_transition_matrix(**kwargs):
    return np.array(
        [[0.14966637, 0.23909554, 0.17954179, 0.13327908, 0.09173705,
          0.10185553, 0.10482464],
         [0.09214617, 0.24221219, 0.34301137, 0.15641113, 0.05539155,
            0.05343969, 0.05738791],
            [0.02441788, 0.07601293, 0.44239171, 0.39349111, 0.0268898,
             0.01866277, 0.0181338],
            [0.02386522, 0.03134441, 0.05317593, 0.65149879, 0.15412951,
             0.05678701, 0.02919912],
            [0.06588236, 0.06176376, 0.04644521, 0.24331156, 0.3150266,
             0.17835363, 0.08921687],
            [0.11647971, 0.11436043, 0.07613927, 0.14996217, 0.20614209,
             0.1988984, 0.13801792],
            [0.15491169, 0.17458328, 0.11415852, 0.13190113, 0.13318385,
             0.15015618, 0.14110536]]
    )
    dataset = get_dataset(**kwargs)
    possible_etas = set(dataset.eta.values.ravel())
    indices_by_eta_dict = {
        eta: np.argwhere(dataset.eta.values == eta)
        for eta in possible_etas
    }
    transition_matrix = []
    for eta_row in range(len(indices_by_eta_dict)):
        transition_row = []
        n_in_quantile = count_total_starting_occurences(
            dataset, indices_by_eta_dict[eta_row])
        for eta_col in possible_etas:
            transition_counts = count_transition_occurences(
                indices_by_eta_dict[eta_row],
                indices_by_eta_dict[eta_col])
            transition_probability = transition_counts / n_in_quantile
            transition_row.append(transition_probability)
        # ensure probabilities sum to 1 (they are never more than 10e-5 off)
        assert abs(1 - sum(transition_row)) < 0.001
        transition_row[eta_row] += (1 - sum(transition_row))
        transition_matrix.append(transition_row)
    return np.array(transition_matrix)
