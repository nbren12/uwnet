from sklearn.externals import joblib
import xarray as xr
from lib.data import prepare_data
inputs = xr.open_dataset(snakemake.input[0])
forcings = xr.open_dataset(snakemake.input[1])
output_data = prepare_data(inputs, forcings)
joblib.dump(output_data, snakemake.output[0])
