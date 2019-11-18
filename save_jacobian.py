from uwnet.wave import LinearResponseFunction, base_from_xarray, model_plus_damping
import xarray as xr
import torch

path = "../../nn/NNLowerDecayLR/20.pkl"
src = torch.load(path)
mean = xr.open_dataset("../../data/processed/training.mean.nc")
eq_mean = mean.isel(y=32)

src = model_plus_damping(src)
base_state = base_from_xarray(eq_mean)
lrf = LinearResponseFunction.from_model(src, base_state)