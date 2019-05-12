from sklearn.model_selection import train_test_split
from scipy import linalg
from copy import copy
import numpy as np
from sklearn.preprocessing import quantile_transform
from uwnet.stochastic_parameterization.utils import (
    get_dataset,
    coarsen_array,
    dataset_dt_seconds,
    default_binning_quantiles,
    default_binning_method,
    default_ds_location,
    default_base_model_location,
    default_eta_transitioner_model,
    default_eta_transitioner_poly_degree,
    default_eta_transitioner_predictors,
    uncoarsen_2d_array,
)


class EtaTransitioner(object):

    def __init__(
            self,
            poly_degree=copy(default_eta_transitioner_poly_degree),
            dt_seconds=copy(dataset_dt_seconds),
            model=copy(default_eta_transitioner_model),
            predictors_to_use=copy(default_eta_transitioner_predictors),
            t_start=0,
            t_stop=640,
            eta_coarsening=None,
            quantile_transform_data=False,
            binning_quantiles=copy(default_binning_quantiles),
            binning_method=copy(default_binning_method),
            ds_location=copy(default_ds_location),
            max_qt_for_residual_model=15,
            multi_model_transitioner=False,
            verbose=True,
            max_sli_for_residual_model=18,
            use_nn_output=False,
            markov_process=True,
            base_model_location=copy(default_base_model_location),
            blur_sigma=None):
        self.verbose = verbose
        self.multi_model_transitioner = multi_model_transitioner
        self.blur_sigma = blur_sigma
        self.eta_coarsening = eta_coarsening
        self.t_start = t_start
        self.t_stop = t_stop
        self.predictors = predictors_to_use
        if use_nn_output:
            self.predictors = self.predictors + [
                'moistening_pred', 'heating_pred'
            ]
        if self.blur_sigma:
            blurred_predictors = []
            for input_ in self.predictors:
                if input_ in ['PW', 'QT', 'SLI']:
                    blurred_predictors.append(input_ + '_blurred')
                else:
                    blurred_predictors.append(input_)
            self.predictors = blurred_predictors
        self.max_qt_for_residual_model = max_qt_for_residual_model
        self.max_sli_for_residual_model = max_sli_for_residual_model
        self.poly_degree = poly_degree
        self.model = model
        self.is_trained = False
        self.ds_location = ds_location
        self.binning_quantiles = binning_quantiles
        self.etas = list(range(len(binning_quantiles)))
        self.dt_seconds = dt_seconds
        self.quantile_transform_data = quantile_transform_data
        self.binning_method = binning_method
        self.base_model_location = base_model_location
        self.set_normalization_params()
        self.markov_process = markov_process

    def transform_transition_matrix_to_timestep(self, transition_matrix):
        if self.dt_seconds != dataset_dt_seconds:
            continuous_transition_matrix = linalg.logm(
                transition_matrix) / dataset_dt_seconds
            return linalg.expm(continuous_transition_matrix * self.dt_seconds)
        return transition_matrix

    def set_normalization_params(self):
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
        self.layer_mass = ds.layer_mass.values
        normalization_params = {}
        for predictor in self.predictors:
            mean_by_degree = {}
            std_by_degree = {}
            data = ds[predictor].values
            if len(data.shape) == 4:
                data = np.average(
                    data,
                    axis=1,
                    weights=ds.layer_mass.values)
            else:
                data = data
            if self.eta_coarsening:
                data = coarsen_array(data, self.eta_coarsening)
            data = data.ravel().reshape(-1, 1)
            for degree in range(1, self.poly_degree + 1):
                mean = (data ** degree).mean(axis=0)
                std = (data ** degree).std()
                mean_by_degree[degree] = mean
                std_by_degree[degree] = std
            normalization_params[predictor] = {
                'mean': mean_by_degree, 'std': std_by_degree
            }
            if '_blurred' in predictor:
                normalization_params[predictor.replace('_blurred', '')] = {
                    'mean': mean_by_degree, 'std': std_by_degree
                }
        self.normalization_params = normalization_params

    def normalize_array(self, array, variable, degree):
        return (
            (array ** degree) -
            self.normalization_params[variable]['mean'][degree]
        ) / self.normalization_params[variable]['std'][degree]

    def format_training_data_one_model(self):
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
        start_times = np.array(range(len(ds.time) - 1))
        stop_times = start_times + 1
        start = ds.isel(time=start_times).eta.values
        y_data = ds.isel(time=stop_times).eta.values
        if self.eta_coarsening:
            y_data = coarsen_array(y_data, self.eta_coarsening)
            start = coarsen_array(start, self.eta_coarsening)
        y_data = y_data.ravel()
        start = start.ravel()
        if self.markov_process:
            x_data = np.zeros((len(start), len(self.etas)))
            x_data[np.arange(len(y_data)), start.astype(int)] = 1
        else:
            x_data = np.zeros((len(start), 0))
        for predictor in self.predictors:
            data = ds.isel(time=start_times)[predictor].values
            if len(data.shape) == 4:
                data = np.average(
                    data, axis=1, weights=ds.layer_mass.values)
            if self.eta_coarsening:
                data = coarsen_array(data, self.eta_coarsening)
            data = data.ravel().reshape(-1, 1)
            for degree in range(1, self.poly_degree + 1):
                x_data = np.hstack([
                    x_data,
                    self.normalize_array(data, predictor, degree)
                ])
        if self.quantile_transform_data:
            x_data = quantile_transform(x_data, axis=0)
        return x_data, y_data

    def format_training_data_multi_model(self):
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
        start_times = np.array(range(len(ds.time) - 1))
        stop_times = start_times + 1
        start = ds.isel(time=start_times).eta.values
        y_data = ds.isel(time=stop_times).eta.values
        if self.eta_coarsening:
            y_data = coarsen_array(y_data, self.eta_coarsening)
            start = coarsen_array(start, self.eta_coarsening)
        y_data = y_data.ravel()
        x_data = start.ravel().reshape(-1, 1)

        for predictor in self.predictors:
            data = ds.isel(time=start_times)[predictor].values
            if len(data.shape) == 4:
                data = np.average(
                    data, axis=1, weights=ds.layer_mass.values)
            if self.eta_coarsening:
                data = coarsen_array(data, self.eta_coarsening)
            data = data.ravel().reshape(-1, 1)
            for degree in range(1, self.poly_degree + 1):
                x_data = np.hstack([
                    x_data,
                    self.normalize_array(data, predictor, degree)
                ])
        if self.quantile_transform_data:
            x_data = quantile_transform(x_data, axis=0)
        return {
            eta: (
                x_data[x_data[:, 0] == eta][:, 1:],
                y_data[x_data[:, 0] == eta]
            )
            for eta in self.etas
        }

    def train_model(self, x_data, y_data):
        x_data, x_test, y_data, y_test = train_test_split(
            x_data, y_data, test_size=0.1)
        model = copy(self.model).fit(x_data, y_data)
        if self.verbose:
            print('\n\nTransitioner Train Score:',
                  f'{model.score(x_data, y_data)}')
            print('\n\nTransitioner Test Score:',
                  f'{model.score(x_test, y_test)}')
        return model

    def train(self):
        if len(self.etas) > 1:
            if self.multi_model_transitioner:
                training_data = self.format_training_data_multi_model()
                models = {}
                for eta in self.etas:
                    x_data, y_data = training_data[eta]
                    models[eta] = self.train_model(x_data, y_data)
                self.transitioner_model = models
            else:
                x_data, y_data = self.format_training_data_one_model()
                self.transitioner_model = self.train_model(x_data, y_data)
        self.is_trained = True

    def get_input_array_from_state_true(self, etas, state):
        print('Warning! This true eta transitioning has not been maintained,',
              'use to efficient eta transitioning.')
        input_array = np.zeros(
            (etas.size * len(self.etas),
             len(self.etas) + (self.poly_degree * len(self.predictors)))
        )
        for eta in range(len(self.etas)):
            rows = range(eta * etas.size, eta * etas.size + etas.size)
            input_array[rows, eta] = 1
            i_ = 0
            for predictor in self.predictors:
                data = state[predictor].numpy()
                if len(data.shape) > 2 and data.shape[0] > 1:
                    data = np.average(
                        data, axis=0, weights=self.layer_mass)
                for degree in range(1, self.poly_degree + 1):
                    input_array[
                        rows,
                        len(self.etas) + i_] = \
                        self.normalize_array(
                            data.ravel(),
                            predictor,
                            degree
                    )
                    i_ += 1
        if self.quantile_transform_data:
            return quantile_transform(input_array, axis=0)
        return input_array

    def get_transition_probabilities_true(self, etas, state):
        input_array = self.get_input_array_from_state_true(etas, state)
        transition_matrices = self.transitioner_model.predict_proba(
            input_array).reshape(
            etas.size, len(self.etas), len(self.etas), order='F')
        if self.dt_seconds != dataset_dt_seconds:
            transition_matrices = np.array([
                self.transform_transition_matrix_to_timestep(
                    transition_matrices[idx])
                for idx in range(len(transition_matrices))
            ])
        return transition_matrices[
            range(len(transition_matrices)), etas.ravel(), :]

    def transition_etas_true(self, etas, state):
        if not self.is_trained:
            raise Exception('Transition Matrix Model not Trained')
        probabilities = self.get_transition_probabilities_true(etas, state)
        c = probabilities.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        return (u < c).argmax(axis=1).reshape(etas.shape)

    def get_input_array_from_state(self, etas, state):
        if self.markov_process:
            input_array = np.zeros((etas.size, len(self.etas)))
            input_array[range(len(input_array)), etas.ravel()] = 1
        else:
            input_array = np.zeros((etas.size, 0))
        i_ = 0
        for predictor in self.predictors:
            data = state[predictor].numpy()
            if len(data.shape) > 2 and data.shape[0] > 1:
                data = np.average(
                    data, axis=0, weights=self.layer_mass)
            if self.eta_coarsening:
                data = coarsen_array(data, self.eta_coarsening)
            data = data.ravel().reshape(-1, 1)
            for degree in range(1, self.poly_degree + 1):
                input_array = np.hstack([
                    input_array,
                    self.normalize_array(data, predictor, degree)
                ])
                i_ += 1
        if self.quantile_transform_data:
            return quantile_transform(input_array, axis=0)
        return input_array

    def get_input_array_from_state_multi_model(self, etas, state):
        input_array = etas.ravel().reshape(-1, 1)
        i_ = 0
        for predictor in self.predictors:
            data = state[predictor].numpy()
            if len(data.shape) > 2 and data.shape[0] > 1:
                data = np.average(
                    data, axis=0, weights=self.layer_mass)
            if self.eta_coarsening:
                data = coarsen_array(data, self.eta_coarsening)
            data = data.ravel().reshape(-1, 1)
            for degree in range(1, self.poly_degree + 1):
                input_array = np.hstack([
                    input_array,
                    self.normalize_array(data, predictor, degree)
                ])
                i_ += 1
        if self.quantile_transform_data:
            return quantile_transform(input_array, axis=0)
        return {
            eta: input_array[input_array[:, 0]][:, 1:] == eta
            for eta in self.etas
        }

    def get_transition_probabilities_efficient_multi_model(
            self, etas, state):
        input_arrays = self.get_input_array_from_state_multi_model(
            etas, state)
        raveled_etas = etas.ravel()
        transition_probabilities = np.zeros(
            (len(raveled_etas), len(self.etas)))
        for eta, input_array in input_arrays.items():
            transition_probs = self.transitioner_model[eta].predict_proba(
                input_array)
            if self.dt_seconds != dataset_dt_seconds:
                ratio = self.dt_seconds / dataset_dt_seconds
                p_stay = transition_probs[:, eta]
                p_transition = 1 - p_stay
                p_transition_new = p_transition * ratio
                p_stay_new = 1 - p_transition_new
                transition_probs = transition_probs * ratio
                transition_probs[:, eta] = p_stay_new
                transition_probabilities[
                    raveled_etas == eta, :] = transition_probs
        return transition_probabilities

    def get_transition_probabilities_efficient(self, etas, state):
        input_array = self.get_input_array_from_state(etas, state)
        transition_probabilities = self.transitioner_model.predict_proba(
            input_array)
        if self.dt_seconds != dataset_dt_seconds:
            ratio = self.dt_seconds / dataset_dt_seconds
            p_stay = transition_probabilities[
                range(len(input_array)), etas.ravel()]
            p_transition = 1 - p_stay
            p_transition_new = p_transition * ratio
            p_stay_new = 1 - p_transition_new
            transition_probabilities = transition_probabilities * ratio
            transition_probabilities[
                range(len(input_array)), etas.ravel()] = p_stay_new
        return transition_probabilities

    def transition_etas_efficient(self, etas, state):
        if not self.is_trained:
            raise Exception('Transition Matrix Model not Trained')
        if self.multi_model_transitioner:
            probabilities = \
                self.get_transition_probabilities_efficient_multi_model(
                    etas, state)
        else:
            probabilities = self.get_transition_probabilities_efficient(
                etas, state)
        c = probabilities.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        return (u < c).argmax(axis=1).reshape(etas.shape)

    def transition_etas(
            self, etas, state, efficient=True, output=None):
        if self.eta_coarsening:
            etas = coarsen_array(etas, self.eta_coarsening).astype(int)
        if output:
            state['moistening_pred'] = output['QT'].detach()
            state['heating_pred'] = output['SLI'].detach()
        if efficient:
            new_etas = self.transition_etas_efficient(etas, state)
            if self.eta_coarsening:
                new_etas = uncoarsen_2d_array(new_etas)
            return new_etas
        return self.transition_etas_true(etas, state)
