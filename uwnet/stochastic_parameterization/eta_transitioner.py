from scipy import linalg
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import quantile_transform
from sklearn.ensemble import GradientBoostingClassifier
from uwnet.stochastic_parameterization.utils import (
    get_dataset,
    dataset_dt_seconds,
    default_binning_quantiles,
    default_binning_method,
    default_ds_location,
    default_base_model_location,
)

default_model = GradientBoostingClassifier(max_depth=500, verbose=2)
default_model = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=10000)
predictors = [
    'SST',
    'PW',
    'QT',
    'SLI',
    # 'FQT',
    # 'FSLI',
    'SHF',
    'LHF',
    'SOLIN',
    # 'RADSFC',
    # 'RADTOA',
    # 'FU',
    # 'FV'
]


class EtaTransitioner(object):

    def __init__(
            self,
            poly_degree=4,
            dt_seconds=dataset_dt_seconds,
            model=default_model,
            predictors=predictors,
            t_start=0,
            t_stop=640,
            quantile_transform_data=False,
            binning_quantiles=default_binning_quantiles,
            binning_method=default_binning_method,
            ds_location=default_ds_location,
            average_z_direction=True,
            max_qt_for_residual_model=15,
            max_sli_for_residual_model=18,
            markov_process=True,
            base_model_location=default_base_model_location):
        self.t_start = t_start
        self.t_stop = t_stop
        self.max_qt_for_residual_model = max_qt_for_residual_model
        self.max_sli_for_residual_model = max_sli_for_residual_model
        self.poly_degree = poly_degree
        self.average_z_direction = average_z_direction
        self.model = model
        self.predictors = predictors
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
            binning_quantiles=self.binning_quantiles,
            binning_method=self.binning_method,
            base_model_location=self.base_model_location
        )
        self.layer_mass = ds.layer_mass.values
        normalization_params = {}
        for predictor in self.predictors:
            mean_by_degree = {}
            std_by_degree = {}
            data = ds[predictor].values
            if len(data.shape) == 4:
                if self.average_z_direction:
                    data = np.average(
                        data, axis=1, weights=ds.layer_mass.values).reshape(
                            -1, 1)
                    data_for_std = data.copy()
                else:
                    data = np.swapaxes(
                        data, 1, 3).reshape((int(data.size / 34), 34))
                    if predictor == 'QT':
                        data = data[:, :self.max_qt_for_residual_model]
                        layer_mass = ds.layer_mass.values[
                            :self.max_qt_for_residual_model]
                    elif predictor == 'SLI':
                        data = data[:, :self.max_sli_for_residual_model]
                        layer_mass = ds.layer_mass.values[
                            :self.max_sli_for_residual_model]
                    else:
                        layer_mass = ds.layer_mass.values
                    data_for_std = data.dot(layer_mass).copy()
            else:
                data = data.ravel().reshape(-1, 1)
                data_for_std = data.copy()
            for degree in range(1, self.poly_degree + 1):
                mean = (data ** degree).mean(axis=0)
                std = (data_for_std ** degree).std()
                mean_by_degree[degree] = mean
                std_by_degree[degree] = std
            normalization_params[predictor] = {
                'mean': mean_by_degree, 'std': std_by_degree
            }
        self.normalization_params = normalization_params

    def normalize_array(self, array, variable, degree):
        return (
            (array ** degree) -
            self.normalization_params[variable]['mean'][degree]
        ) / self.normalization_params[variable]['std'][degree]

    def format_training_data(self):
        ds = get_dataset(
            ds_location=self.ds_location,
            t_start=self.t_start,
            t_stop=self.t_stop,
            binning_quantiles=self.binning_quantiles,
            binning_method=self.binning_method,
            base_model_location=self.base_model_location
        )
        start_times = np.array(range(len(ds.time) - 1))
        stop_times = start_times + 1
        start = ds.isel(time=start_times).eta.values.ravel()
        y_data = ds.isel(time=stop_times).eta.values.ravel()
        if self.markov_process:
            x_data = np.zeros((len(start), len(self.etas)))
            x_data[np.arange(len(y_data)), start] = 1
        else:
            x_data = np.zeros((len(start), 0))
        for predictor in self.predictors:
            data = ds.isel(time=start_times)[predictor].values
            if len(data.shape) == 4:
                if self.average_z_direction:
                    data_for_predictor = np.average(
                        data, axis=1, weights=ds.layer_mass.values).reshape(
                            -1, 1)
                else:
                    data_for_predictor = np.swapaxes(
                        data, 1, 3).reshape((len(y_data), 34))
                    if predictor == 'QT':
                        data_for_predictor = data_for_predictor[
                            :, :self.max_qt_for_residual_model]
                    elif predictor == 'SLI':
                        data_for_predictor = data_for_predictor[
                            :, :self.max_sli_for_residual_model]
            else:
                data_for_predictor = data.ravel().reshape(-1, 1)
            for degree in range(1, self.poly_degree + 1):
                x_data = np.hstack([
                    x_data,
                    self.normalize_array(data_for_predictor, predictor, degree)
                ])
        if self.quantile_transform_data:
            x_data = quantile_transform(x_data, axis=0)
        return x_data, y_data

    def train(self):
        if len(self.etas) > 1:
            x_data, y_data = self.format_training_data()
            from sklearn.model_selection import train_test_split
            x_data, x_test, y_data, y_test = train_test_split(
                x_data, y_data, test_size=0.6)
            self.model.fit(x_data, y_data)
        self.is_trained = True

    def get_input_array_from_state_true(self, etas, state):
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
        transition_matrices = self.model.predict_proba(
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
                if self.average_z_direction:
                    data = np.average(
                        data, axis=0, weights=self.layer_mass).ravel().reshape(
                            -1, 1)
                else:
                    data = np.swapaxes(data, 0, 2).reshape(
                        (len(input_array), 34))
                    if predictor == 'QT':
                        data = data[:, :self.max_qt_for_residual_model]
                    elif predictor == 'SLI':
                        data = data[:, :self.max_sli_for_residual_model]
            else:
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

    def get_transition_probabilities_efficient(self, etas, state):
        input_array = self.get_input_array_from_state(etas, state)
        transition_probabilities = self.model.predict_proba(
            input_array)
        if self.dt_seconds != dataset_dt_seconds:
            ratio = self.dt_seconds / dataset_dt_seconds
            p_stay = transition_probabilities[
                range(len(input_array)), etas.ravel()]
            transition_probabilities = transition_probabilities * ratio
            transition_probabilities[
                range(len(input_array)), etas.ravel()] = 1 - (
                    (1 - p_stay) * ratio)
        return transition_probabilities

    def transition_etas_efficient(self, etas, state):
        if not self.is_trained:
            raise Exception('Transition Matrix Model not Trained')
        probabilities = self.get_transition_probabilities_efficient(
            etas, state)
        c = probabilities.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        return (u < c).argmax(axis=1).reshape(etas.shape)

    def transition_etas(self, etas, state, efficient=True):
        if efficient:
            return self.transition_etas_efficient(etas, state)
        return self.transition_etas_true(etas, state)
