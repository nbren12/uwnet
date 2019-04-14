import numpy as np
from functools import lru_cache
import torch
from uwnet.tensordict import TensorDict
from uwnet.thermo import compute_apparent_source
import xarray as xr

model_dir = ''
# model_dir = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'  # noqa
model_location = model_dir + 'stochastic_model.pkl'
default_base_model_location = model_dir + 'full_model/1.pkl'
default_ds_location = "training.nc"
# default_ds_location = "uwnet/stochastic_parameterization/training.nc"
dataset_dt_seconds = 10800
# default_binning_method = 'precip'
default_binning_method = 'column_integrated_qt_residuals'
# default_binning_method = 'column_integrated_sli_residuals'
default_binning_quantiles = (0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1)
# default_binning_quantiles = (.1, .3, .7, .9, 1)
# default_binning_quantiles = (1,)
# default_binning_quantiles = (.01, .05, .15, .35, .5, .65, .85, .95, .99, 1)


class BaseModel(object):

    def __init__(
            self,
            model_location,
            dataset,
            binning_quantiles=default_binning_quantiles,
            time_step_days=.125,
            binning_method=default_binning_method):
        self.model = torch.load(model_location)
        self.ds = dataset
        self.time_step_days = time_step_days
        self.time_step_seconds = 86400 * time_step_days
        self.binning_quantiles = binning_quantiles
        self.binning_method = binning_method

    def get_true_forcings(self):
        qt_forcing = compute_apparent_source(
            self.ds.QT,
            self.ds.FQT * 86400)
        sli_forcing = compute_apparent_source(
            self.ds.SLI,
            self.ds.FSLI * 86400)
        return xr.Dataset({'QT': qt_forcing, 'SLI': sli_forcing})

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
        column_integrated_qt_residuals = np.ones_like(self.ds.Prec.values)
        column_integrated_sli_residuals = np.ones_like(self.ds.Prec.values)
        moistening_residuals = np.ones_like(self.ds.QT.values)
        heating_residuals = np.ones_like(self.ds.SLI.values)
        moistening_preds = np.ones_like(self.ds.QT.values)
        heating_preds = np.ones_like(self.ds.SLI.values)
        time_indices = list(range(len(self.ds.time) - 1))
        true_forcings = self.get_true_forcings()
        idx = 0
        for time_batch in np.array_split(time_indices, 10):
            idx += 1
            pred = self.predict_for_time_idx(time_batch)
            true_qt = true_forcings.isel(time=time_batch).QT
            true_sli = true_forcings.isel(time=time_batch).SLI
            q2_preds = np.ma.average(
                pred['QT'],
                axis=1,
                weights=self.ds.layer_mass.values
            )
            q2_true = np.ma.average(
                true_qt,
                axis=1,
                weights=self.ds.layer_mass.values
            )
            column_integrated_qt_residuals[
                time_batch, :, :] = q2_true - q2_preds
            q1_preds = np.ma.average(
                pred['SLI'],
                axis=1,
                weights=self.ds.layer_mass.values
            )
            q1_true = np.ma.average(
                true_sli,
                axis=1,
                weights=self.ds.layer_mass.values
            )
            column_integrated_sli_residuals[
                time_batch, :, :] = q1_true - q1_preds

            moistening_preds[time_batch, :, :, :] = pred['QT']
            heating_preds[time_batch, :, :, :] = pred['SLI']
            moistening_residuals[time_batch, :, :, :] = true_qt - pred['QT']
            heating_residuals[time_batch, :, :, :] = true_sli - pred['SLI']
        self.heating_preds = heating_preds
        self.moistening_preds = moistening_preds
        self.column_integrated_qt_residuals = column_integrated_qt_residuals
        self.column_integrated_sli_residuals = column_integrated_sli_residuals
        self.moistening_residuals = moistening_residuals
        self.heating_residuals = heating_residuals

    def get_bin_membership(self):
        self.get_qt_ratios()
        if self.binning_method == 'precip':
            bins = [
                np.quantile(self.ds.Prec.values, quantile)
                for quantile in self.binning_quantiles
            ]
            return np.digitize(
                self.ds.Prec.values, bins, right=True)
        elif self.binning_method == 'column_integrated_qt_residuals':
            bins = [
                np.quantile(self.column_integrated_qt_residuals, quantile)
                for quantile in self.binning_quantiles
            ]
            return np.digitize(
                self.column_integrated_qt_residuals, bins, right=True)
        elif self.binning_method == 'column_integrated_sli_residuals':
            bins = [
                np.quantile(self.column_integrated_sli_residuals, quantile)
                for quantile in self.binning_quantiles
            ]
            return np.digitize(
                self.column_integrated_sli_residuals, bins, right=True)
        raise Exception(f'Binning method {self.binning_method} not recognized')


def insert_nn_output_precip_ratio_bin_membership(
        dataset,
        binning_quantiles,
        base_model_location,
        binning_method):
    base_model = BaseModel(
        base_model_location,
        dataset,
        binning_quantiles=binning_quantiles,
        binning_method=binning_method
    )
    bin_membership = base_model.get_bin_membership()
    eta_ = dataset['Prec'].copy()
    eta_.values = bin_membership
    dataset['eta'] = eta_
    dataset['eta'].attrs['units'] = ''
    dataset['eta'].attrs['long_name'] = 'Stochastic State'
    dataset['column_integrated_qt_residuals'] = dataset['Prec'].copy()
    dataset['column_integrated_qt_residuals'].values = \
        base_model.column_integrated_qt_residuals
    dataset['nn_moistening_residual'] = dataset['QT'].copy()
    dataset['nn_moistening_residual'].values = base_model.moistening_residuals
    dataset['nn_heating_residual'] = dataset['SLI'].copy()
    dataset['nn_heating_residual'].values = base_model.heating_residuals
    dataset['moistening_pred'] = dataset['QT'].copy()
    dataset['moistening_pred'].values = base_model.moistening_preds
    dataset['heating_pred'] = dataset['SLI'].copy()
    dataset['heating_pred'].values = base_model.heating_preds
    del base_model
    return dataset


def get_xarray_dataset_with_eta(
        data,
        binning_quantiles,
        base_model_location=None,
        t_start=0,
        t_stop=640,
        set_eta=True,
        binning_method=default_binning_method):
    try:
        dataset = xr.open_zarr(data)
    except ValueError:
        dataset = xr.open_dataset(data)
    dataset = dataset.isel(time=range(t_start, t_stop))
    if set_eta:
        dataset = insert_nn_output_precip_ratio_bin_membership(
            dataset, binning_quantiles, base_model_location, binning_method)
    try:
        return dataset.isel(step=0).drop('step').drop('p')
    except Exception:
        return dataset


@lru_cache()
def get_dataset(
        ds_location=default_ds_location,
        base_model_location=default_base_model_location,
        add_precipital_water=True,
        t_start=0,
        t_stop=640,
        set_eta=True,
        binning_quantiles=default_binning_quantiles,
        binning_method=default_binning_method):
    ds = get_xarray_dataset_with_eta(
        ds_location,
        binning_quantiles,
        base_model_location=base_model_location,
        t_start=t_start,
        t_stop=t_stop,
        set_eta=set_eta,
        binning_method=binning_method
    )
    if add_precipital_water:
        ds['PW'] = (ds.QT * ds.layer_mass).sum('z') / 1000
    return ds


def count_transition_occurences(starting_indices, ending_indices):
    time_step_after_start = starting_indices.copy()
    time_step_after_start[:, 0] += 1
    set_a = set([tuple(coords) for coords in time_step_after_start])
    set_b = set([tuple(coords) for coords in ending_indices])
    return len(set_a.intersection(set_b))


def count_total_starting_occurences(dataset, indices_by_eta):
    max_time_idx = len(dataset.time) - 1
    return (indices_by_eta[:, 0] != max_time_idx).sum()
