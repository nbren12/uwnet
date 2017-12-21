from lib.models.torch.preprocess import prepare_data
from sklearn.externals import joblib

input_files = snakemake.input.inputs
forcing_files = snakemake.input.forcing
weight_file = snakemake.input.weight

output_data = prepare_data(input_files, forcing_files, weight_file)
joblib.dump(output_data, snakemake.output[0])
