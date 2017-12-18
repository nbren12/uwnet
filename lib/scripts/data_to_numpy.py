import numpy as np
from lib.preprocess import prepare_data

def main():
    input_files = snakemake.input.inputs
    forcing_files = snakemake.input.forcing
    weight_file = snakemake.input.weight

    output_data = prepare_data(input_files, forcing_files, weight_file)

    np.savez(snakemake.output[0], **output_data)

if __name__ == '__main__':
    main()

# forcing_files = [
#     "data/calc/forcing/ngaqua/sl.nc",
#     "data/calc/forcing/ngaqua/qt.nc",
# ]

# input_files = ["data/calc/ngaqua/sl.nc", "data/calc/ngaqua/qt.nc"]

# weight_file = "data/processed/ngaqua/w.nc"


# output_data = prepare_data(input_files, forcing_files, weight_file)
# from IPython import embed; embed()
