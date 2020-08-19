import os
import sys
import pickle
import common

import numpy as np
import xarray as xr

SECONDS_PER_DAY = 86400
KG_KG_TO_G_KG = 1000
Q1_SCALE = SECONDS_PER_DAY
Q2_SCALE = SECONDS_PER_DAY * KG_KG_TO_G_KG
TRUTH_NAME = "truth"

def midpoint(x):
    return (x[1:] + x[:-1]) / 2


def read_2d_data(url, nn):
    NN_NAME = nn
    S = common.load_pickle_from_url(url)

    # Coordinates
    coords = {"path_bins": midpoint(S["QMspace"]), "lts_bins": midpoint(S["LTSspace"])}

    data_vars = {}
    # Histogram quantities (Figure 2)
    dims_hist = ["path_bins", "lts_bins"]
    data_vars["net_precipitation_nn"] = (dims_hist, S["PREChist"][NN_NAME])
    data_vars["net_precipitation_src"] = (dims_hist, S["PREChist"][TRUTH_NAME])
    data_vars["net_heating_nn"] = (dims_hist, S["HEAThist"][NN_NAME])
    data_vars["net_heating_src"] = (dims_hist, S["HEAThist"][TRUTH_NAME])
    data_vars["count"] = (dims_hist, S["Whist"])

    # Vertical quantities Figure 3 and 4

    return xr.Dataset(data_vars, coords=coords)


url, nn, output = sys.argv[1:]
output_dataset = read_2d_data(url, nn)
output_dataset.to_netcdf(output)
