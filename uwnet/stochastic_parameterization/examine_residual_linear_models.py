from functools import lru_cache
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from uwnet.stochastic_parameterization.residual_stochastic_state_model import (  # noqa
    StochasticStateModel,
)
from uwnet.stochastic_parameterization.utils import get_dataset

model_dir = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization'
model_location = model_dir + '/residual_stochastic_model.pkl'
base_model_location = model_dir + '/full_model/1.pkl'
model = torch.load(model_location)


@lru_cache()
def get_average_of_inputs(eta):
    averages = {}
    ds = get_dataset(
        t_start=50,
        t_stop=75)
    ds = ds.where(ds.eta == eta)
    inputs = ['moistening_pred', 'heating_pred'] + model.residual_model_inputs
    for feature_ in inputs:
        averages[feature_] = ds[feature_].mean(['x', 'y', 'time'])


def examine_nn_prediction_coefficients():
    for eta in model.possible_etas:
        input_averages = get_average_of_inputs(eta)
        for idx, prog in enumerate(model.prognostics):
            if prog == 'QT':
                input_feature = 'moistening_pred'
            else:
                input_feature = 'heating_pred'
            coefficients = model.residual_models_by_eta[eta][prog].coef_[
                :, idx * 34: (idx + 1) * 34]
            sns.heatmap(
                coefficients,
                cmap='coolwarm',
                robust=True
            )
            plt.title(
                f'{prog} Coefficients for NN predictions for eta = {eta}')
            plt.ylabel(f'Output {prog} dimenstion')
            plt.xlabel(f'Input {input_feature} dimension')
            plt.show()


if __name__ == '__main__':
    examine_nn_prediction_coefficients()
