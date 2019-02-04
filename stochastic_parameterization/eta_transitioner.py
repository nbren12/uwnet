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
                multi_class='multinomial', solver='lbfgs', max_iter=1000)):
        self.sst_poly_degree = sst_poly_degree
        self.model = model
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
        self.sst_mean = ds.isel(time=0).SST.values.ravel().mean()
        self.sst_std = ds.isel(time=0).SST.values.ravel().std()

    def normalize_sst_array(self, sst_array):
        return (sst_array - self.sst_mean) / self.sst_std

    def get_sst_data(self, times):
        ds = get_dataset()
        sst = ds.isel(time=times).SST.values.ravel()
        sst = np.array([
            self.normalize_sst_array(sst) ** degree
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
        if self.sst_poly_degree:
            sst_data = self.get_sst_data(start_times)
            x_data = np.hstack((x_data, sst_data))
        return x_data, y_data

    def train(self):
        x_data, y_data = self.format_training_data()
        from sklearn.model_selection import train_test_split
        x_data, x_test, y_data, y_test = train_test_split(
            x_data, y_data, test_size=0.9)
        self.model.fit(x_data, y_data)
        self.is_trained = True

    def predict_for_sst(self, sst):
        if not self.is_trained:
            raise Exception('Transition Matrix Model not Trained')
        if self.sst_poly_degree:
            additional_input = [
                self.normalize_sst_array(sst) ** degree
                for degree in range(1, self.sst_poly_degree + 1)
            ]
        else:
            additional_input = []
        transition_matrix = []
        for eta in self.etas:
            input_ = np.zeros(len(self.etas))
            input_[eta] = 1
            input_ = np.concatenate([input_, additional_input]).reshape(1, -1)
            transition_matrix.append(self.model.predict_proba(input_)[0])
        return self.transform_transition_matrix_to_timestep(
            np.array(transition_matrix))

    def transition_eta(self, eta, sst):
        transition_probabilities = self.predict_for_sst(sst)[eta]
        return np.random.choice(self.etas, p=transition_probabilities)

    def transition_etas(self, etas, ssts):
        input_array = np.zeros(
            (etas.size * len(self.etas),
             len(self.etas) + self.sst_poly_degree)
        )
        for eta in range(len(self.etas)):
            rows = range(eta * etas.size, eta * etas.size + etas.size)
            input_array[rows, eta] = 1
            for poly_degree in range(1, self.sst_poly_degree + 1):
                input_array[
                    rows,
                    len(self.etas) + poly_degree - 1] = (
                        self.normalize_sst_array(ssts.ravel()) ** poly_degree)
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
