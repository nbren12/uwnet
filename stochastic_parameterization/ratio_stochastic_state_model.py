import numpy as np
import torch
from stochastic_parameterization.eta_transitioner import EtaTransitioner
from stochastic_parameterization.utils import (
    binning_quantiles,
    get_dataset,
)
from sklearn.linear_model import LinearRegression
from uwnet.sam_interface import get_model
from uwnet.tensordict import TensorDict
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
            training_size=100000,
            prognostics=['QT', 'SLI'],
            ratio_model_inputs=['SST', 'PW', 'QT', 'SLI'],
            base_model_location=base_model_location,
            ratio_model_class=LinearRegression):
        super(StochasticStateModel, self).__init__()
        self.is_trained = False
        self.dims = dims
        self.prognostics = prognostics
        self.dt_seconds = dt_seconds
        self.training_size = training_size
        self.possible_etas = list(range(len(binning_quantiles)))
        self.setup_eta()
        self.ratio_model_class = ratio_model_class
        self.base_model_location = base_model_location
        self.base_model = torch.load(base_model_location)
        # self.setup_eta_transitioner()
        self.ratio_model_inputs = ratio_model_inputs
        self.y_indices = np.array(
            [[idx] * dims[1] for idx in range(dims[0])])

    def setup_eta_transitioner(self):
        transitioner = EtaTransitioner(
            dt_seconds=self.dt_seconds, binning_method='q2_ratio')
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

    def format_training_data_for_ratio_model(self, eta, ds):
        indices = np.argwhere(ds.eta.values == self.eta)
        y_data = np.hstack([
            ds.nn_moistening_ratio.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]],
            ds.nn_heating_ratio.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]]
        ])
        x_data = np.hstack([
            ds.moistening_pred.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]],
            ds.heating_pred.values[
                indices[:, 0], :, indices[:, 1], indices[:, 2]]
        ])
        for variable in self.ratio_model_inputs:
            if len(ds[variable].shape) == 4:
                data_for_var = ds[variable].values[
                    indices[:, 0], :, indices[:, 1], indices[:, 2]
                ]
            else:
                data_for_var = ds[variable].values[
                    indices[:, 0], indices[:, 1], indices[:, 2]
                ].reshape(-1, 1)
            x_data = np.hstack([x_data, data_for_var])
        return x_data, y_data

    def train(self):
        if not self.is_trained:
            ds = get_dataset(
                binning_method='q2_ratio',
                t_start=50,
                t_stop=75)
            residual_ratio_models = {}
            print('Training ratio stochastic state model')
            for eta in self.possible_etas:
                print(f'Training eta={eta}...')
                x_data, y_data = self.format_training_data_for_ratio_model(
                    eta, ds)
                if len(x_data) > self.training_size:
                    sample = np.random.choice(
                        range(len(x_data)),
                        size=self.training_size,
                        replace=False)
                else:
                    sample = range(len(x_data))
                ratio_model = self.ratio_model_class()
                ratio_model.fit(x_data[sample], y_data[sample])
                residual_ratio_models[eta] = ratio_model
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
        output = TensorDict({
            key: torch.zeros_like(x[key]) for key in self.prognostics
        })
        base_model_pred = self.base_model(x)
        for eta, model in self.residual_ratio_models.items():
            indices = np.argwhere(self.eta == eta)
            predictions = model.predict(x)
            # pred_by_prognostic =
            for key in self.prognostics:
                output[key][
                    :, indices[:, 0], indices[:, 1]
                ] = predictions[key][:, indices[:, 0], indices[:, 1]].double()
        return output


def train_a_model():
    model = StochasticStateModel()
    model.train()
    torch.save(model, 'stochastic_parameterization/ratio_stochastic_model.pkl')


if __name__ == '__main__':
    train_a_model()
    config = {
        'type': 'neural_network',
        'path': 'stochastic_parameterization/ratio_stochastic_model.pkl'
    }
    model = get_model(config)
    data = torch.load('/Users/stewart/Desktop/state.pt')
    pred = model(data)
    print(pred)
