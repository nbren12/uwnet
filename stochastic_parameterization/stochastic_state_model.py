import os
import numpy as np
import torch
from stochastic_parameterization.eta_transitioner import EtaTransitioner
from stochastic_parameterization.utils import binning_quantiles
from uwnet.sam_interface import get_model
from uwnet.tensordict import TensorDict
from torch import nn

dataset_dt_seconds = 10800
model_dir = '/Users/stewart/projects/uwnet/stochastic_parameterization'
base_model_location = model_dir + '/full_model/1.pkl'


class StochasticStateModel(nn.Module):

    def __init__(
            self,
            dims=(64, 128),
            dt_seconds=10800,
            prognostics=['QT', 'SLI'],
            base_model_location=base_model_location):
        super(StochasticStateModel, self).__init__()
        self.is_trained = False
        self.dims = dims
        self.prognostics = prognostics
        self.dt_seconds = dt_seconds
        self.possible_etas = list(range(len(binning_quantiles)))
        self.setup_eta()
        self.base_model_location = base_model_location
        self.transition_matrix_model = self.setup_eta_transitioner()
        self.y_indices = np.array(
            [[idx] * dims[1] for idx in range(dims[0])])

    def setup_eta_transitioner(self):
        transitioner = EtaTransitioner(dt_seconds=self.dt_seconds)
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

    def train_conditional_model(
            self,
            eta,
            training_config_file,
            **kwargs):
        cmd = f'python -m uwnet.train with {training_config_file}'
        cmd += f' eta_to_train={eta}'
        cmd += f' output_dir=models/stochastic_state_model_adagrad_{eta}'
        # cmd += f" base_model_location='{self.base_model_location}'"
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
                # self.train_conditional_model(
                #     eta, training_config_file, **kwargs)
                conditional_models[eta] = torch.load(
                    f'models/stochastic_state_model_adagrad_{eta}/1.pkl'
                )
            self.conditional_models = conditional_models
            self.is_trained = True
        else:
            raise Exception('Model already trained')

    def update_eta(self, x):
        self.eta = self.eta_transitioner.transition_etas(self.eta, x)

    def forward(self, x):
        self.update_eta(x)
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
    # data = torch.load('/Users/stewart/Desktop/state.pt')
    # pred = model(data)
    # print(pred)
