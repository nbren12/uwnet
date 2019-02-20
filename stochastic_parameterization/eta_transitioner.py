from scipy import linalg
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from stochastic_parameterization.utils import (
    get_dataset,
    dataset_dt_seconds,
    binning_quantiles,
)

default_model = GradientBoostingClassifier(max_depth=500, verbose=2)
default_model = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=10000)
predictors = [
    'SST',
    'PW',
    'QT',
    'SLI',
    'FQT',
    'FSLI',
    # 'SHF',
    # 'LHF',
    # 'SOLIN',
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
            binning_method='precip'):
        self.poly_degree = poly_degree
        self.model = model
        self.predictors = predictors
        self.binning_method = binning_method
        self.is_trained = False
        self.set_normalization_params()
        self.etas = list(range(len(binning_quantiles)))
        self.dt_seconds = dt_seconds

    def transform_transition_matrix_to_timestep(self, transition_matrix):
        if self.dt_seconds != dataset_dt_seconds:
            continuous_transition_matrix = linalg.logm(
                transition_matrix) / dataset_dt_seconds
            return linalg.expm(continuous_transition_matrix * self.dt_seconds)
        return transition_matrix

    def set_normalization_params(self):
        ds = get_dataset(
            binning_method=self.binning_method,
            t_start=50,
            t_stop=75)
        self.layer_mass = ds.layer_mass.values
        normalization_params = {}
        for predictor in self.predictors:
            mean_by_degree = {}
            std_by_degree = {}
            for degree in range(1, self.poly_degree + 1):
                data = ds[predictor].values
                if len(data.shape) == 4:
                    data = np.average(
                        data, axis=1, weights=ds.layer_mass.values)
                mean = (ds[predictor].values.ravel() ** degree).mean()
                std = (ds[predictor].values.ravel() ** degree).std()
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
            binning_method=self.binning_method,
            t_start=50,
            t_stop=75)
        start_times = np.array(range(len(ds.time) - 1))
        stop_times = start_times + 1
        start = ds.isel(time=start_times).eta.values.ravel()
        y_data = ds.isel(time=stop_times).eta.values.ravel()
        x_data = np.zeros((len(start), len(self.etas)))
        x_data[np.arange(len(y_data)), start] = 1
        for predictor in self.predictors:
            data = ds.isel(time=start_times)[predictor].values
            if len(data.shape) == 4:
                data = np.average(
                    data, axis=1, weights=ds.layer_mass.values)
            for degree in range(1, self.poly_degree + 1):
                data_for_predictor = self.normalize_array(
                    data.ravel(),
                    predictor,
                    degree)
                x_data = np.append(
                    x_data, data_for_predictor.reshape(-1, 1), 1)
        return x_data, y_data

    def train(self):
        x_data, y_data = self.format_training_data()
        from sklearn.model_selection import train_test_split
        x_data, x_test, y_data, y_test = train_test_split(
            x_data, y_data, test_size=0.9)
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
                if data.shape[0] > 1:
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
        return input_array

    def transition_etas_true(self, etas, state):
        if not self.is_trained:
            raise Exception('Transition Matrix Model not Trained')
        input_array = self.get_input_array_from_state_true(etas, state)
        transition_matrices = self.model.predict_proba(
            input_array).reshape(
            etas.size, len(self.etas), len(self.etas), order='F')
        if self.dt_seconds != dataset_dt_seconds:
            transition_matrices = np.vectorize(
                self.transform_transition_matrix_to_timestep)(
                    transition_matrices)
        probabilities = transition_matrices[
            range(len(transition_matrices)), etas.ravel(), :]
        c = probabilities.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        return (u < c).argmax(axis=1).reshape(etas.shape)

    def get_input_array_from_state(self, etas, state):
        input_array = np.zeros(
            (
                etas.size,
                len(self.etas) + (self.poly_degree * len(self.predictors))
            )
        )
        input_array[range(len(input_array)), etas.ravel()] = 1
        i_ = 0
        for predictor in self.predictors:
            data = state[predictor].numpy()
            if len(data.shape) > 2 and data.shape[0] > 1:
                data = np.average(
                    data, axis=0, weights=self.layer_mass)
            for degree in range(1, self.poly_degree + 1):
                input_array[
                    range(len(input_array)),
                    len(self.etas) + i_] = \
                    self.normalize_array(
                        data.ravel(),
                        predictor,
                        degree
                )
                i_ += 1
        return input_array

    def transition_etas_efficient(self, etas, state):
        input_array = self.get_input_array_from_state(etas, state)
        transition_probabilities = self.model.predict_proba(
            input_array)
        if self.dt_seconds != dataset_dt_seconds:
            ratio = self.dt_seconds / dataset_dt_seconds
            p_stay = transition_probabilities[range(len(input_array)), etas]
            transition_probabilities = transition_probabilities * ratio
            transition_probabilities[range(len(input_array)), etas] = 1 - (
                (1 - p_stay) * ratio)
        c = transition_probabilities.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        return (u < c).argmax(axis=1).reshape(etas.shape)

    def transition_etas(self, etas, state, efficient=True):
        if efficient:
            return self.transition_etas_efficient(etas, state)
        return self.transition_etas_true(etas, state)
