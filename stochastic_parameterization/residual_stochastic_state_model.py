import numpy as np
import torch
from stochastic_parameterization.eta_transitioner import EtaTransitioner
from stochastic_parameterization.utils import (
    binning_quantiles,
    get_dataset,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from uwnet.sam_interface import get_model
from torch import nn

dataset_dt_seconds = 10800
model_dir = '/Users/stewart/projects/uwnet/stochastic_parameterization'
base_model_location = model_dir + '/full_model/1.pkl'
residual_ratio_model_path = model_dir + '/residual_ratio_models.pkl'


class StochasticStateModel(nn.Module):

    def __init__(
            self,
            dims=(64, 128),
            dt_seconds=10800,
            prognostics=['QT', 'SLI'],
            ratio_model_inputs=['SST', 'QT', 'SLI'],
            base_model_location=base_model_location,
            ratio_model_class=LinearRegression):
        super(StochasticStateModel, self).__init__()
        self.is_trained = False
        self.dims = dims
        self.prognostics = prognostics
        self.dt_seconds = dt_seconds
        self.possible_etas = list(range(len(binning_quantiles)))
        self.setup_eta()
        self.ratio_model_class = ratio_model_class
        self.base_model_location = base_model_location
        self.base_model = torch.load(base_model_location)
        self.setup_eta_transitioner()
        self.ratio_model_inputs = ratio_model_inputs
        self.y_indices = np.array(
            [[idx] * dims[1] for idx in range(dims[0])])
        self.binning_method = 'q2_residual'

    def setup_eta_transitioner(self):
        transitioner = EtaTransitioner(
            dt_seconds=self.dt_seconds, binning_method='q2_residual')
        transitioner.train()
        self.eta_transitioner = transitioner

    def setup_eta(self):
        self.eta = np.random.choice(
            self.possible_etas,
            self.dims,
            p=np.ediff1d([0] + list(binning_quantiles))
        )

    def eval(self):
        if not self.is_trained:
            raise Exception('Model is not trained')

    def format_x_data_for_ratio_model(self, eta, preds, x, indices):
        x_data = {
            'QT': preds['QT'][:, indices[:, 0], indices[:, 1]],
            'SLI': preds['SLI'][:, indices[:, 0], indices[:, 1]]
        }
        for variable in self.ratio_model_inputs:
            data_for_var = x[variable][:, indices[:, 0], indices[:, 1]]
            for var in ['QT', 'SLI']:
                x_data[var] = torch.cat([x_data[var], data_for_var.float()])
        for var in ['QT', 'SLI']:
            x_data[var] = x_data[var].detach().numpy().T
        return x_data

    def format_training_data_for_ratio_model(self, indices, ds):
        y_data = {
            'QT': ds.nn_moistening_residual.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]],
            'SLI': ds.nn_heating_residual.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]]
        }
        x_data = {
            'QT': ds.moistening_pred.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]],
            'SLI': ds.heating_pred.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]]
        }
        for variable in self.ratio_model_inputs:
            if len(ds[variable].shape) == 4:
                data_for_var = ds[variable].values[
                    indices[:, 0], :, indices[:, 1], indices[:, 2]
                ]
            else:
                data_for_var = ds[variable].values[
                    indices[:, 0], indices[:, 1], indices[:, 2]
                ].reshape(-1, 1)
            for var in ['QT', 'SLI']:
                x_data[var] = np.hstack([x_data[var], data_for_var])
        return x_data, y_data

    def train(self):
        if not self.is_trained:
            ds = get_dataset(
                binning_method='q2_residual',
                t_start=50,
                t_stop=75)
            residual_ratio_models = {}
            print('Training ratio stochastic state model')
            for eta in self.possible_etas:
                print(f'Training eta={eta}...')
                indices = np.argwhere(ds.eta.values == eta)
                x_data, y_data = self.format_training_data_for_ratio_model(
                    indices, ds)
                ratio_models = {}
                for var in ['QT', 'SLI']:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x_data[var], y_data[var], test_size=0.5)
                    ratio_model = self.ratio_model_class()
                    ratio_model.fit(x_train, y_train)
                    test_score = ratio_model.score(x_test, y_test)
                    train_score = ratio_model.score(x_train, y_train)
                    print(f'{var} test score: {test_score}')
                    print(f'{var} train score: {train_score}')
                    ratio_models[var] = ratio_model
                residual_ratio_models[eta] = ratio_models
            self.residual_ratio_models = residual_ratio_models
            self.is_trained = True
        else:
            raise Exception('Model already trained')

    def update_eta(self, x):
        self.eta = self.eta_transitioner.transition_etas(self.eta, x)

    def forward(self, x, eta=None):
        if eta is not None:
            self.eta = eta
        else:
            self.update_eta(x)
        output = self.base_model(x)
        for eta, model in self.residual_ratio_models.items():
            indices = np.argwhere(self.eta == eta)
            x_data = self.format_x_data_for_ratio_model(
                eta, output, x, indices)
            for key in self.prognostics:
                output[
                    key][:, indices[:, 0], indices[:, 1]] += torch.from_numpy(
                        model[key].predict(x_data[key]).T)
        return output


def train_a_model():
    model = StochasticStateModel()
    model.train()
    torch.save(
        model, 'stochastic_parameterization/residual_stochastic_model.pkl')


if __name__ == '__main__':
    train_a_model()
    config = {
        'type': 'neural_network',
        'path': 'stochastic_parameterization/ratio_stochastic_model.pkl'
    }
    model = get_model(config)
    data = torch.load('/Users/stewart/Desktop/state.pt')
    pred = model(data)
    # print(pred)
