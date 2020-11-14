"""Convert netCDF dataset to a memory efficient layout for machine learning

The original dataset should have the dimensions ('time', 'z', 'y', 'x'). This
script combines the y and x dimensions into a sample dimension with a specified
chunk size. This sample dimension is optionally shufffled, and then saved to a
zarr archive.

The saved archive is suitable to use with uwnet.train
"""
import xarray as xr
import numpy as np

# arguments
input_file = snakemake.input[0]
output_file = snakemake.output[0]
variables = snakemake.params.variables
shuffle = snakemake.params.shuffle
train_or_test = snakemake.wildcards.train_or_test

chunk_size = 2**10

# open data
ds = xr.open_dataset(input_file)


if train_or_test == "train":
    ds = ds.isel(x=slice(0, 64))
elif train_or_test == "test":
    ds = ds.isel(x=slice(64, None))
else:
    raise NotImplementedError("{test_or_train} is not \"train\" or \"test\"")

# perform basic validation
assert ds['SLI'].dims == ('time', 'z', 'y', 'x')

# stack data
variables = list(variables) # needs to be a list for xarray
stacked = (ds[variables]
           .stack(sample=['y', 'x'])
           .drop('sample'))

# add needed variables
stacked['layer_mass'] = ds.layer_mass.isel(time=0)

# shuffle samples
if shuffle:
    n = len(stacked.sample)
    indices = np.random.choice(n, n, replace=False)
    stacked = stacked.isel(sample=indices)

chunked = stacked.chunk({'sample': chunk_size})

# save to disk
chunked.to_zarr(output_file)
