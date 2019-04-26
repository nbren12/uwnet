# coding: utf-8
from callpy import get_state_as_dataset
import torch
import climt
import sympl

state = torch.load("assets/state.pt")
ds = get_state_as_dataset(state)
ds = ds.rename({'x_wind': 'eastward_wind', 'y_wind': 'northward_wind'})
ds = {key: sympl.DataArray(ds[key]) for key in ds.data_vars}
ds['time'] = 0.0

hs = climt.HeldSuarez()

tend, diag = hs(ds)
