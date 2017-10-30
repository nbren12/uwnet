from sklearn.externals import joblib
from xnoah.data_matrix import stack_cat
from lib.util import compute_weighted_scale, weights_to_np, scales_to_np

import xarray as xr


def get_dmd_data(input_files, forcing_files):

    inputs = xr.open_mfdataset(input_files)
    forcings = xr.open_mfdataset(forcing_files)

    # only use tropics
    inputs = inputs.isel(y=slice(24, 40))
    forcings = forcings.isel(y=slice(24, 40))

    # inputs are everything but the final time step
    X = inputs.isel(time=slice(0, -1))
    dt = float(inputs.time[1]-inputs.time[0])
    # Apply one time step of the external forcings to X
    X = X + dt * forcings.isel(time=slice(0,-1))
    # outputs are the time shifted inputs
    Y = inputs.shift(time=-1)\
            .isel(time=slice(0,-1))

    return X.load(), Y.load()


def train_test_split(X, f=.6):
    ntrain = int(X['time'].shape[0] * f)

    return X.isel(time=slice(0, ntrain)),\
        X.isel(time=slice(ntrain, -1))




def prepvar(X, feature_dims=['z'], sample_dims=['time', 'x', 'y']):
    # select only the tropics
    return stack_cat(X, "features", ['z'])\
        .stack(samples=['time', 'x', 'y'])\
        .transpose("samples", "features")


def get_processed_dmd_data(input_files, forcing_files, weight_file):

    # get weights
    w = xr.open_dataarray(weight_file)
    # open data and perform train test split
    X, Y = get_dmd_data(input_files, forcing_files)

    x_train, x_test = train_test_split(X, f=.7)
    y_train, y_test = train_test_split(Y, f=.7)

    # flatten the data
    flat_data = [prepvar(x) for x in [x_train, y_train, x_test, y_test]]
    # TODO this is hard coded and should be checked for compatibility with
    # output of prepvar
    sample_dims = ['time', 'x', 'y'] 
    feature_dims= ['z']

    # compute variable scales
    scales_in = compute_weighted_scale(w, sample_dims=sample_dims, ds=x_train)
    scales_out = compute_weighted_scale(w, sample_dims=sample_dims, ds=y_train)

    # make scales and weights vectors
    x_train, y_train, x_test, y_test = flat_data

    scales_in = scales_to_np(scales_in, x_train.indexes['features'])
    scales_out = scales_to_np(scales_out, y_train.indexes['features'])

    # get input and output weights
    w_input = weights_to_np(w, x_train.indexes['features'])
    w_output = weights_to_np(w, y_train.indexes['features'])


    # dict of outputs
    output_data = {
        'weight': (w_input, w_output),
        'scale': (scales_in, scales_out),
        'train': flat_data[:2],
        'test': flat_data[2:]
    }

    return output_data

def main():
    input_files = snakemake.input.inputs
    forcing_files = snakemake.input.forcing
    weight_file = snakemake.input.weight

    output_data = get_processed_dmd_data(input_files, forcing_files, weight_file)

    joblib.dump(output_data, snakemake.output[0])

if __name__ == '__main__':
    main()

# forcing_files = [
#     "data/calc/forcing/ngaqua/sl.nc",
#     "data/calc/forcing/ngaqua/qt.nc",
# ]

# input_files = ["data/calc/ngaqua/sl.nc", "data/calc/ngaqua/qt.nc"]

# weight_file = "data/processed/ngaqua/w.nc"


# output_data = get_processed_dmd_data(input_files, forcing_files, weight_file)
# from IPython import embed; embed()
