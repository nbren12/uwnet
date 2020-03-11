import os
import pickle

import numpy as np
import xarray as xr

SECONDS_PER_DAY = 86400
KG_KG_TO_G_KG = 1000
Q1_SCALE = SECONDS_PER_DAY
Q2_SCALE = SECONDS_PER_DAY * KG_KG_TO_G_KG
PATH_PKL = "data/PKL_DATA/"
NN_NAME = "STAB"
TRUTH_NAME = "truth"


def midpoint(x):
    return (x[1:] + x[:-1]) / 2


def read_2d_data():
    filename = "9_30_Fig2_" + NN_NAME + ".pkl"
    path = os.path.join(PATH_PKL, filename)
    S = np.load(path, allow_pickle=True)

    with open(path, "rb") as hf:
        S = pickle.load(hf)

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


def read_3d_data():

    filename = "9_30_Fig3_" + NN_NAME + ".pkl"
    path_to_vertical_data = os.path.join(PATH_PKL, filename)
    path_to_coords = os.path.join(PATH_PKL, "11_2_SPCAM_coordinates.pkl")

    with open(path_to_coords, "rb") as f:
        Scoor = pickle.load(f)

    with open(path_to_vertical_data, "rb") as f:
        S34 = pickle.load(f)

    # build dataset
    data_vars = {}
    dims_3d = ["path_bins", "lts_bins", "z"]

    data_vars["Q1NN"] = (dims_3d, S34["dTdthist"][NN_NAME] * Q1_SCALE)
    data_vars["Q2NN"] = (dims_3d, S34["dqdthist"][NN_NAME] * Q2_SCALE)

    data_vars["Q1"] = (
        dims_3d,
        S34["dTdthist"][TRUTH_NAME] * Q1_SCALE
    )
    data_vars["Q2"] = (
        dims_3d,
        S34["dqdthist"][TRUTH_NAME] * Q2_SCALE
    )

    data_vars["QV"] = (dims_3d, S34["QVhist"] *  KG_KG_TO_G_KG)
    data_vars["TABS"] = (dims_3d, S34["Thist"])

    # TODO Ask tom if scoor['lev'] as actually the pressure
    data_vars["p"] = (["z"], np.asarray(Scoor["lev"]))

    return xr.Dataset(data_vars)


output_dataset = xr.merge([read_3d_data(), read_2d_data()])
output_dataset.to_netcdf("data/tom_binned.nc")
