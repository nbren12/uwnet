from stochastic_parameterization.stochastic_state_model import (
    StochasticStateModel,
    dataset_dt_seconds,
)
from unittest.mock import patch
import numpy as np

mock_transition_matrix = np.array([.5] * 4).reshape(2, 2)


class MockConditionalModel():

    def __init__(self, ret_val):
        self.ret_val = ret_val

    def forward(self, x):
        return self.ret_val


def setup_model():
    mod = StochasticStateModel()
    mod.transition_matrix = mock_transition_matrix
    mod.possible_etas = list(range(len(mock_transition_matrix)))
    mod.dims = (3, 3)
    mod.dt_seconds = dataset_dt_seconds
    mod.precip_quantiles = [.5, 1]
    return mod


def test_setup_transition_matrix():
    with patch.object(StochasticStateModel, '__init__', lambda x: None):
        mod = setup_model()
        mod.setup_transition_matrix()
        assert (mod.transition_matrix.sum(axis=1) != 1).sum() == 0


def test_setup_transition_matrix_smaller_dt():
    with patch.object(StochasticStateModel, '__init__', lambda x: None):
        mod = setup_model()
        mod.setup_transition_matrix()
        initial_transition_matrix = mod.transition_matrix
        mod.dt_seconds = int(mod.dt_seconds / 10)
        mod.setup_transition_matrix()
        for i in range(len(mod.transition_matrix)):
            assert mod.transition_matrix[i][i] > initial_transition_matrix[
                i][i]


def test_setup_eta():
    with patch.object(StochasticStateModel, '__init__', lambda x: None):
        mod = setup_model()
        mod.setup_eta()
        assert hasattr(mod, 'eta')
        assert mod.eta.shape == mod.dims


def test_update_eta():
    with patch.object(StochasticStateModel, '__init__', lambda x: None):
        mod = setup_model()
        mod.eta = np.zeros(mod.dims)
        mod.transition_matrix = np.zeros_like(mod.transition_matrix)
        mod.transition_matrix[:, 1] = 1
        mod.update_eta()
        assert (mod.eta != 1).sum() == 0


def test_predict():
    with patch.object(StochasticStateModel, '__init__', lambda x: None):
        mod = setup_model()
        mod.setup_eta()
        mod.conditional_models = {
            eta: MockConditionalModel(eta) for eta in mod.possible_etas
        }
        mod.is_trained = True
        x = np.zeros(mod.dims)
        preds = mod.predict(x)
        assert preds.shape == mod.dims
        for x in range(mod.dims[0]):
            for y in range(mod.dims[1]):
                assert preds[x, y] == mod.eta[x, y]
