import numpy as np
from copy import copy
from sklearn.preprocessing import quantile_transform
import torch
import xarray as xr
from uwnet.stochastic_parameterization.eta_transitioner import EtaTransitioner
from uwnet.stochastic_parameterization.utils import (
    default_binning_quantiles,
    default_binning_method,
    get_dataset,
    model_dir,
    dataset_dt_seconds,
    default_ds_location,
    default_base_model_location,
    default_eta_transitioner_predictors,
)
from uwnet.xarray_interface import XRCallMixin
from uwnet.tensordict import TensorDict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.serialization import SourceChangeWarning
import warnings

warnings.filterwarnings("ignore", category=SourceChangeWarning)

default_t_start = 100
default_t_stop = 150
model_inputs = ['SST', 'QT', 'SLI', 'SOLIN']
default_quantile_transform_data = False


class StochasticStateModel(nn.Module, XRCallMixin):

    def __init__(
            self,
            dims=(64, 128),
            dt_seconds=10800,
            eta_transitioner_dt_seconds=10800,
            prognostics=['QT', 'SLI'],
            residual_model_inputs=model_inputs,
            max_sli_for_residual_model=18,
            max_qt_for_residual_model=15,
            t_start=copy(default_t_start),
            t_stop=copy(default_t_stop),
            blur_sigma=None,
            eta_coarsening=None,
            residual_model_class=LinearRegression,
            binning_quantiles=copy(default_binning_quantiles),
            binning_method=copy(default_binning_method),
            ds_location=copy(default_ds_location),
            base_model_location=copy(default_base_model_location),
            quantile_transform_data=copy(default_quantile_transform_data),
            eta_predictors=copy(default_eta_transitioner_predictors),
            include_output_in_transition_model=True,
            time_idx_to_use_for_eta_initialization='random',
            markov_process=True,
            change_blurred_var_names=True,
            verbose=True):
        super(StochasticStateModel, self).__init__()
        self.t_start = t_start
        self.t_stop = t_stop
        self.change_blurred_var_names = change_blurred_var_names
        self.blur_sigma = blur_sigma
        self.eta_coarsening = eta_coarsening
        self.eta_predictors = eta_predictors
        self.binning_quantiles = binning_quantiles
        self.include_output_in_transition_model = \
            include_output_in_transition_model
        self.markov_process = markov_process
        self.binning_method = binning_method
        self.base_model_location = base_model_location
        self.verbose = verbose
        self.quantile_transform_data = quantile_transform_data
        self.ds_location = ds_location
        self.is_trained = False
        self.dims = dims
        self.max_sli_for_residual_model = max_sli_for_residual_model
        self.max_qt_for_residual_model = max_qt_for_residual_model
        self.prognostics = prognostics
        self.possible_etas = list(range(len(binning_quantiles)))
        self._dt_seconds = dt_seconds
        self.eta_transitioner_dt_seconds = eta_transitioner_dt_seconds
        if eta_transitioner_dt_seconds != dt_seconds:
            warnings.warn(
                'Eta Transitioner and Stochastic Model have differnet dts')
        self.setup_eta(time_idx_to_use_for_eta_initialization)
        self.residual_model_class = residual_model_class
        self.base_model = torch.load(base_model_location)
        self.setup_eta_transitioner()
        self.residual_model_inputs = residual_model_inputs
        if self.blur_sigma:
            blurred_inputs = []
            for input_ in self.residual_model_inputs:
                if input_ in ['PW', 'QT', 'SLI']:
                    blurred_inputs.append(input_ + '_blurred')
                else:
                    blurred_inputs.append(input_)
            self.residual_model_inputs = blurred_inputs

    @property
    def dt_seconds(self):
        return self._dt_seconds

    @dt_seconds.setter
    def dt_seconds(self, dt_seconds):
        self._dt_seconds = dt_seconds
        if hasattr(self, 'eta_transitioner'):
            self.eta_transitioner.dt_seconds = dt_seconds

    def setup_eta_transitioner(self):
        transitioner = EtaTransitioner(
            ds_location=self.ds_location,
            dt_seconds=self.eta_transitioner_dt_seconds,
            t_start=self.t_start,
            t_stop=self.t_stop,
            verbose=self.verbose,
            eta_coarsening=self.eta_coarsening,
            use_nn_output=self.include_output_in_transition_model,
            quantile_transform_data=self.quantile_transform_data,
            binning_quantiles=self.binning_quantiles,
            binning_method=self.binning_method,
            markov_process=self.markov_process,
            max_qt_for_residual_model=self.max_qt_for_residual_model,
            max_sli_for_residual_model=self.max_sli_for_residual_model,
            predictors_to_use=self.eta_predictors,
            base_model_location=self.base_model_location,
            blur_sigma=self.blur_sigma)
        self.eta_transitioner = transitioner

    def setup_eta(self, time_idx_to_use_for_eta_initialization=0):
        ds = get_dataset(
            ds_location=self.ds_location,
            t_start=self.t_start,
            t_stop=self.t_stop,
            eta_coarsening=self.eta_coarsening,
            binning_quantiles=self.binning_quantiles,
            binning_method=self.binning_method,
            base_model_location=self.base_model_location,
            blur_sigma=self.blur_sigma
        )
        if time_idx_to_use_for_eta_initialization == 'random':
            time_idx_to_use_for_eta_initialization = np.random.choice(
                np.arange(len(ds.time)))
        self.eta = ds.isel(
            time=time_idx_to_use_for_eta_initialization).eta.values

    def simulate_eta(
            self,
            t_start=None,
            n_time_steps=50):
        if t_start is None:
            t_start = self.t_start
        self.setup_eta()
        ds = get_dataset(
            ds_location=self.ds_location,
            t_start=t_start,
            t_stop=t_start + n_time_steps,
            eta_coarsening=self.eta_coarsening,
            binning_quantiles=self.binning_quantiles,
            binning_method=self.binning_method,
            base_model_location=self.base_model_location,
            blur_sigma=self.blur_sigma)
        for time in range(len(ds.time)):
            input_data = {}
            for predictor in self.eta_transitioner.predictors:
                input_data[predictor] = torch.from_numpy(
                    ds.isel(time=time)[predictor].values)
            self.update_eta(TensorDict(input_data))

    def eval(self):
        if not self.is_trained:
            raise Exception('Model is not trained')

    def format_x_data_for_residual_model(self, eta, preds, x, indices):
        pred_input = torch.cat([
            preds['QT'][
                :self.max_qt_for_residual_model,
                indices[:, 0],
                indices[:, 1]],
            preds['SLI'][
                :self.max_sli_for_residual_model,
                indices[:, 0],
                indices[:, 1]],
        ])
        x_data = {'QT': pred_input, 'SLI': pred_input}
        for variable in self.residual_model_inputs:
            if 'QT' in variable:
                data_for_var = x[variable][
                    :self.max_qt_for_residual_model,
                    indices[:, 0],
                    indices[:, 1]]
            elif 'SLI' in variable:
                data_for_var = x[variable][
                    :self.max_sli_for_residual_model,
                    indices[:, 0],
                    indices[:, 1]]
            else:
                data_for_var = x[variable][:, indices[:, 0], indices[:, 1]]
            for var in ['QT', 'SLI']:
                x_data[var] = torch.cat([x_data[var], data_for_var.float()])
        for var in ['QT', 'SLI']:
            if self.quantile_transform_data:
                x_data[var] = quantile_transform(
                    x_data[var].detach().numpy().T, axis=0)
            else:
                x_data[var] = x_data[var].detach().numpy().T
        return x_data

    def format_training_data_for_residual_model(self, indices, ds):
        y_data = {
            'QT': ds.nn_moistening_residual.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]],
            'SLI': ds.nn_heating_residual.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]]
        }
        preds = np.hstack([
            ds.moistening_pred.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]
            ][:, :self.max_qt_for_residual_model],
            ds.heating_pred.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]
            ][:, :self.max_sli_for_residual_model]
        ])
        x_data = {'QT': preds, 'SLI': preds}
        for variable in self.residual_model_inputs:
            if 'QT' in variable:
                data_for_var = ds[variable].values[
                    indices[:, 0],
                    :self.max_qt_for_residual_model,
                    indices[:, 1],
                    indices[:, 2]
                ]
            elif 'SLI' in variable:
                data_for_var = ds[variable].values[
                    indices[:, 0],
                    :self.max_sli_for_residual_model,
                    indices[:, 1],
                    indices[:, 2]
                ]
            elif len(ds[variable].shape) == 4:
                data_for_var = ds[variable].values[
                    indices[:, 0], :, indices[:, 1], indices[:, 2]
                ]
            else:
                data_for_var = ds[variable].values[
                    indices[:, 0], indices[:, 1], indices[:, 2]
                ].reshape(-1, 1)
            for var in ['QT', 'SLI']:
                x_data[var] = np.hstack([x_data[var], data_for_var])
        if self.quantile_transform_data:
            for var in ['QT', 'SLI']:
                x_data[var] = quantile_transform(x_data[var], axis=0)
        return x_data, y_data

    def train(self):
        if not self.is_trained:
            self.eta_transitioner.train()
            ds = get_dataset(
                ds_location=self.ds_location,
                t_start=self.t_start,
                t_stop=self.t_stop,
                eta_coarsening=None,
                binning_quantiles=self.binning_quantiles,
                binning_method=self.binning_method,
                base_model_location=self.base_model_location,
                blur_sigma=self.blur_sigma)
            residual_models_by_eta = {}
            if self.verbose:
                print('Training residual stochastic state model')
            for eta in self.possible_etas:
                if self.verbose:
                    print(f'Training eta={eta}...')
                indices = np.argwhere(ds.eta.values == eta)
                x_data, y_data = self.format_training_data_for_residual_model(
                    indices, ds)
                residual_models = {}
                for var in ['QT', 'SLI']:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x_data[var], y_data[var], test_size=0.5)
                    residual_model = self.residual_model_class()
                    residual_model.fit(x_train, y_train)
                    test_score = residual_model.score(x_test, y_test)
                    train_score = residual_model.score(x_train, y_train)
                    if self.verbose:
                        print(f'{var} test score: {test_score}')
                        print(f'{var} train score: {train_score}')
                    residual_models[var] = residual_model
                residual_models_by_eta[eta] = residual_models
            self.residual_models_by_eta = residual_models_by_eta
            self.is_trained = True
            if self.change_blurred_var_names and self.blur_sigma:
                unblurred_vars = []
                for variable in self.residual_model_inputs:
                    if '_blurred' in variable:
                        unblurred_vars.append(variable.replace('_blurred', ''))
                    else:
                        unblurred_vars.append(variable)
                self.residual_model_inputs = unblurred_vars
                unblurred_transitioner_vars = []
                for variable in self.eta_transitioner.predictors:
                    if '_blurred' in variable:
                        unblurred_transitioner_vars.append(
                            variable.replace('_blurred', ''))
                    else:
                        unblurred_transitioner_vars.append(variable)
                    self.eta_transitioner.predictors = \
                        unblurred_transitioner_vars
        else:
            raise Exception('Model already trained')

    def update_eta(self, x, output=None):
        if len(self.binning_quantiles) > 1:
            self.eta = self.eta_transitioner.transition_etas(
                self.eta, x, output=output)

    def forward(self, x, eta=None, return_stochastic_state=True):
        # return {
        #     'QT': torch.zeros(34, 64, 128),
        #     'SLI': torch.zeros(34, 64, 128),
        # }
        if (('PW' in self.residual_model_inputs) or (
                'PW' in self.eta_transitioner.predictors)) and 'PW' not in x:
            x['PW'] = (x['QT'] * x['layer_mass'].reshape(
                34, 1, 1)).sum(0) / 1000
        output = self.base_model(x)
        if eta is not None:
            self.eta = eta
        else:
            if self.include_output_in_transition_model:
                self.update_eta(x, output)
            else:
                self.update_eta(x)
        for eta, model in self.residual_models_by_eta.items():
            indices = np.argwhere(self.eta == eta)
            if len(indices) > 0:
                x_data = self.format_x_data_for_residual_model(
                    eta, output, x, indices)
                for key in self.prognostics:
                    output[key][:, indices[:, 0], indices[:, 1]] += (
                        self.dt_seconds / dataset_dt_seconds) * (
                            torch.from_numpy(
                                model[key].predict(x_data[key]).T).float())
        if return_stochastic_state:
            output['stochastic_state'] = torch.from_numpy(self.eta)
        return output

    def predict(self, x, eta=None):
        outputs = []
        for time in x.time:
            row_for_time = x.sel(time=time)
            output_for_time = self.call_with_xr(
                row_for_time,
                eta=eta,
                return_stochastic_state=False)
            outputs.append(output_for_time)
        return xr.concat(outputs, dim='time')


def train_a_model():
    model = StochasticStateModel()
    model.train()
    torch.save(
        model,
        model_dir + 'residual_stochastic_model.pkl')
