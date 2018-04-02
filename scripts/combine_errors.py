import numpy as np
import re
import json
import xarray as xr
import pandas as pd


def read_train_loss(epoch, fname,
                    variables=['test_loss', 'train_loss']):
    """Read the loss.json file for the current epochs test and train loss"""
    df = pd.read_json(fname)
    epoch_means = df.groupby('epoch').mean()

    # need to look for epoch-1 because this data is accumulated over the whole first epoch
    if epoch > 0:
        return epoch_means.loc[epoch-1][variables].to_dict()
    else:
        return {'test_loss': np.nan, 'train_loss': np.nan}


errors = []
dims = []
pattern = re.compile("data/output/model.(.*?)/(.*?)/(.*?)/error.nc")
for f in snakemake.input:
    m = pattern.search(f)
    if m:
        model, seed, epoch = m.groups()
        ds = xr.open_dataset(f)

        arg_file = f"data/output/model.{model}/{seed}/arguments.json"
        args = json.load(open(arg_file))
        # nhidden is a list, so need to just take the first element
        # since all the neural networks I fit are single layer
        args['nhidden'] = args['nhidden'][0]
        args.pop('seed', None)
        ds = ds.assign(**args)

        loss_file = f"data/output/model.{model}/{seed}/loss.json"
        train_error = read_train_loss(int(epoch), loss_file)
        ds = ds.assign(**train_error)

        # append to lists
        dims.append((model, seed, int(epoch)))
        errors.append(ds)


names = ['model', 'seed', 'epoch']
dim = pd.MultiIndex.from_tuples(dims, names=names)
dim.name = 'tmp'
ds = xr.concat(errors, dim=dim).unstack('tmp')

ds.to_netcdf(snakemake.output[0])
