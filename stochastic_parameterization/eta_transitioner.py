from scipy import linalg
import numpy as np
from sklearn.linear_model import LogisticRegression
from stochastic_parameterization.utils import (
    get_dataset,
    dataset_dt_seconds,
    binning_quantiles,
)


class EtaTransitioner(object):

    def __init__(
            self,
            sst_poly_degree=3,
            dt_seconds=dataset_dt_seconds,
            model=LogisticRegression(
                multi_class='multinomial', solver='lbfgs', max_iter=1000),
            predictors=['SST', 'PW']):
        self.sst_poly_degree = sst_poly_degree
        self.model = model
        self.predictors = predictors
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
        ds = get_dataset()
        normalization_params = {}
        for predictor in self.predictors:
            mean = ds.isel(time=0)[predictor].values.ravel().mean()
            std = ds.isel(time=0)[predictor].values.ravel().std()
            normalization_params[predictor] = {
                'mean': mean, 'std': std
            }
        self.normalization_params = normalization_params

    def normalize_array(self, array, variable):
        return (array - self.normalization_params[variable]['mean']
                ) / self.normalization_params[variable]['std']

    def get_sst_data(self, times):
        ds = get_dataset()
        sst = ds.isel(time=times).SST.values.ravel()
        sst = np.array([
            self.normalize_array(sst, 'SST') ** degree
            for degree in range(1, self.sst_poly_degree + 1)
        ])
        return np.stack(sst).T

    def format_training_data(self):
        ds = get_dataset()
        start_times = np.array(range(len(ds.time) - 1))
        stop_times = start_times + 1
        start = ds.isel(time=start_times).eta.values.ravel()
        y_data = ds.isel(time=stop_times).eta.values.ravel()
        x_data = np.zeros((len(start), len(self.etas)))
        x_data[np.arange(len(y_data)), start] = 1
        if 'SST' in self.predictors and self.sst_poly_degree:
            sst_data = self.get_sst_data(start_times)
            x_data = np.hstack((x_data, sst_data))
        for predictor in self.predictors:
            if predictor != 'SST':
                data_for_predictor = self.normalize_array(
                    ds.isel(time=start_times)[
                        predictor].values.ravel(), predictor)
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

    def transition_etas(self, etas, state):
        if not self.is_trained:
            raise Exception('Transition Matrix Model not Trained')
        input_array = np.zeros(
            (etas.size * len(self.etas),
             len(self.etas) + self.sst_poly_degree + len(self.predictors) - 1)
        )
        for eta in range(len(self.etas)):
            rows = range(eta * etas.size, eta * etas.size + etas.size)
            input_array[rows, eta] = 1
            for poly_degree in range(1, self.sst_poly_degree + 1):
                input_array[
                    rows,
                    len(self.etas) + poly_degree - 1
                ] = (
                    self.normalize_array(
                        state['SST'].numpy().ravel(),
                        'SST'
                    ) ** poly_degree)
            i_ = 0
            for predictor in self.predictors:
                if predictor != 'SST':
                    input_array[
                        rows,
                        len(self.etas) + self.sst_poly_degree + i_] = \
                            self.normalize_array(
                                state[predictor].numpy().ravel(), predictor)
                    i_ += 1
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
