from uwnet.train import get_xarray_dataset
import torch
from stochastic_parameterization.stochastic_state_model import (  # noqa
    StochasticStateModel,
)

model_location = '/Users/stewart/projects/uwnet/stochastic_parameterization/stochastic_model.pkl'  # noqa

model = torch.load(model_location)
ds = get_xarray_dataset(
    "/Users/stewart/projects/uwnet/data/processed/training.nc",
    model.precip_quantiles
)
