import numpy as np
import torch
from uwnet.stochastic_parameterization.eta_transitioner import EtaTransitioner
from uwnet.stochastic_parameterization.utils import (
    binning_quantiles,
    get_dataset,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from torch import nn

dataset_dt_seconds = 10800
# model_dir = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'  # noqa
model_dir = ''
base_model_location = model_dir + 'full_model/1.pkl'

t_start = 100
t_stop = 150
model_inputs = ['SST', 'QT', 'SLI', 'SOLIN']


class StochasticStateModel(nn.Module):

    def __init__(
            self,
            dims=(64, 128),
            dt_seconds=10800,
            prognostics=['QT', 'SLI'],
            residual_model_inputs=model_inputs,
            base_model_location=base_model_location,
            max_sli_for_residual_model=18,
            max_qt_for_residual_model=15,
            residual_model_class=LinearRegression):
        super(StochasticStateModel, self).__init__()
        self.is_trained = False
        self.dims = dims
        self.max_sli_for_residual_model = max_sli_for_residual_model
        self.max_qt_for_residual_model = max_qt_for_residual_model
        self.prognostics = prognostics
        self.possible_etas = list(range(len(binning_quantiles)))
        self._dt_seconds = dt_seconds
        self.setup_eta()
        self.residual_model_class = residual_model_class
        self.base_model_location = base_model_location
        self.base_model = torch.load(base_model_location)
        self.setup_eta_transitioner()
        self.residual_model_inputs = residual_model_inputs

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
            dt_seconds=self.dt_seconds,
            t_start=t_start,
            t_stop=t_stop)
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
            data_for_var = x[variable][:, indices[:, 0], indices[:, 1]]
            for var in ['QT', 'SLI']:
                x_data[var] = torch.cat([x_data[var], data_for_var.float()])
        for var in ['QT', 'SLI']:
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
            ds = get_dataset(t_start=t_start, t_stop=t_stop)
            residual_models_by_eta = {}
            print('Training residual stochastic state model')
            for eta in self.possible_etas:
                print(f'Training eta={eta}...')
                indices = np.argwhere(ds.eta.values == eta)
                x_data, y_data = self.format_training_data_for_residual_model(
                    indices, ds)
                residual_models = {}
                for var in ['QT', 'SLI']:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x_data[var], y_data[var], test_size=0.2)
                    residual_model = self.residual_model_class()
                    residual_model.fit(x_train, y_train)
                    test_score = residual_model.score(x_test, y_test)
                    train_score = residual_model.score(x_train, y_train)
                    print(f'{var} test score: {test_score}')
                    print(f'{var} train score: {train_score}')
                    residual_models[var] = residual_model
                residual_models_by_eta[eta] = residual_models
            self.residual_models_by_eta = residual_models_by_eta
            self.is_trained = True
        else:
            raise Exception('Model already trained')

    def update_eta(self, x):
        if len(binning_quantiles) > 1:
            self.eta = self.eta_transitioner.transition_etas(self.eta, x)

    def forward(self, x, eta=None):
        if (('PW' in self.residual_model_inputs) or (
                'PW' in self.eta_transitioner.predictors)) and 'PW' not in x:
            x['PW'] = (x['QT'] * x['layer_mass'].reshape(
                34, 1, 1)).sum(0) / 1000
        if eta is not None:
            self.eta = eta
        else:
            self.update_eta(x)
        output = self.base_model(x)
        for eta, model in self.residual_models_by_eta.items():
            indices = np.argwhere(self.eta == eta)
            x_data = self.format_x_data_for_residual_model(
                eta, output, x, indices)
            for key in self.prognostics:
                output[key][:, indices[:, 0], indices[:, 1]] += (
                    self.dt_seconds / dataset_dt_seconds) * torch.from_numpy(
                        model[key].predict(x_data[key]).T).float()
        output['stochastic_state'] = torch.from_numpy(self.eta)
        return output


def train_a_model():
    model = StochasticStateModel()
    model.train()
    torch.save(
        model,
        'uwnet/stochastic_parameterization/residual_stochastic_model.pkl')
