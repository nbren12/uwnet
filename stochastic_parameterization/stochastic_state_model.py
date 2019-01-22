import os
import numpy as np
import torch
from scipy import linalg
from stochastic_parameterization.utils import (
    get_q2_ratio_transition_matrix,
)
from uwnet.sam_interface import get_model
from uwnet.tensordict import TensorDict
from torch import nn

dataset_dt_seconds = 10800
model_dir = '/Users/stewart/projects/uwnet/stochastic_parameterization'
base_model_location = model_dir + '/full_model/1.pkl'

class StochasticStateModel(nn.Module):

    def __init__(
            self,
            binning_quantiles=[0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1],
            dims=(64, 128),
            dt_seconds=10800,
            prognostics=['QT', 'SLI'],
            base_model_location=base_model_location):
        super(StochasticStateModel, self).__init__()
        self.is_trained = False
        self.dims = dims
        self.prognostics = prognostics
        self.dt_seconds = dt_seconds
        self.binning_quantiles = binning_quantiles
        self.possible_etas = list(range(len(binning_quantiles)))
        self.setup_eta()
        self.transition_matrix = get_q2_ratio_transition_matrix(
            binning_quantiles=self.binning_quantiles)
        self.base_model_location = base_model_location
        self.setup_transition_matrix()

    def setup_transition_matrix(self):
        if self.dt_seconds != dataset_dt_seconds:
            continuous_transition_matrix = linalg.logm(
                self.transition_matrix) / dataset_dt_seconds
            self.transition_matrix = linalg.expm(
                continuous_transition_matrix * self.dt_seconds)

    def setup_eta(self):
        self.eta = np.random.choice(
            self.possible_etas,
            self.dims,
            p=np.ediff1d([0] + list(self.binning_quantiles))
        )

    def eval(self):
        if not self.is_trained:
            raise Exception('Model is not trained')

    def train_conditional_model(
            self,
            eta,
            training_config_file,
            **kwargs):
        cmd = f'python -m uwnet.train with {training_config_file}'
        cmd += f' eta_to_train={eta}'
        cmd += f' output_dir=models/stochastic_state_model_q2_ratio_{eta}'
        cmd += f" binning_quantiles='{self.binning_quantiles}'"
        cmd += f" prognostics='{self.prognostics}'"
        cmd += f" base_model_location='{self.base_model_location}'"
        for key, val in kwargs.items():
            cmd += f' {key}={val}'
        os.system(cmd)

    def train(
            self,
            training_config_file='assets/training_configurations/default.json',
            **kwargs):
        conditional_models = {}
        if not self.is_trained:
            for eta in self.possible_etas:
                self.train_conditional_model(
                    eta, training_config_file, **kwargs)
                conditional_models[eta] = torch.load(
                    f'models/stochastic_state_model_q2_ratio_{eta}/1.pkl'
                )
            self.conditional_models = conditional_models
            self.is_trained = True
        else:
            raise Exception('Model already trained')

    def update_eta(self):
        new_eta = np.zeros_like(self.eta)
        for eta in self.possible_etas:
            indices = np.argwhere(self.eta == eta)
            next_etas = np.random.choice(
                self.possible_etas,
                len(indices),
                p=self.transition_matrix[eta]
            )
            new_eta[indices[:, 0], indices[:, 1]] = next_etas
        self.eta = new_eta

    def forward(self, x):
        self.update_eta()
        output = TensorDict({
            key: torch.zeros_like(x[key]) for key in self.prognostics
        })
        for eta, model in self.conditional_models.items():
            indices = np.argwhere(self.eta == eta)
            predictions = model(x)
            for key in self.prognostics:
                output[key][
                    :, indices[:, 0], indices[:, 1]
                ] = predictions[key][:, indices[:, 0], indices[:, 1]].double()
        return output


def train_a_model():
    model = StochasticStateModel()
    kwargs = {'epochs': 1}
    model.train(**kwargs)
    torch.save(model, 'stochastic_parameterization/stochastic_model.pkl')


if __name__ == '__main__':
    train_a_model()
    config = {
        'type': 'neural_network',
        'path': 'stochastic_parameterization/stochastic_model.pkl'
    }
    model = get_model(config)
    data = torch.load('/Users/stewart/Desktop/state.pt')
    pred = model(data)
    print(pred)
