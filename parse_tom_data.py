import os
import pickle

import xarray as xr


def midpoint(x):
    return (x[1:] + x[:-1]) / 2


pathPKL = "data/PKL_DATA/"
NNname = "STAB"
truth_name = 'truth'
filename = "9_30_Fig2_" + NNname + ".pkl"
path = os.path.join(pathPKL, filename)

with open(path, "rb") as hf:
    S = pickle.load(hf)

# Coordinates
coords = {"path_bins": midpoint(S["QMspace"]), "lts_bins": midpoint(S["LTSspace"])}

data_vars = {}
# Histogram quantities (Figure 2)
dims_hist = ['path_bins', 'lts_bins']
data_vars['net_precipitation_nn'] = (dims_hist, S['PREChist'][NNname])
data_vars['net_precipitation_src'] = (dims_hist, S['PREChist'][truth_name])
data_vars['net_heating_nn'] = (dims_hist, S['HEAThist'][NNname])
data_vars['net_heating_src'] = (dims_hist, S['HEAThist'][truth_name])
data_vars['count'] = (dims_hist, S['Whist'])

# Vertical quantities

output_dataset = xr.Dataset(data_vars, coords=coords)
output_dataset.to_netcdf("data/tom_binned.nc")
